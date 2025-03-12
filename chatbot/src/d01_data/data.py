import os
import json
from pinecone import ServerlessSpec
import time
from tqdm import tqdm

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

def read_json(filename):
    """
    Lee un archivo JSON y devuelve su contenido.
    
    Parámetros:
    filename (str): Ruta del archivo JSON a leer.

    Retorna:
    dict o list: Contenido del archivo JSON como un diccionario o lista.
    """
    try:
        with open(filename, 'r') as archivo:
            contenido = json.load(archivo)
        return contenido
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no existe.")
    except json.JSONDecodeError:
        print(f"Error: El archivo '{filename}' no es un JSON válido.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


def json_dump(datos, nombre_archivo):
    """
    Guarda un objeto JSON en un archivo .json.

    Parámetros:
    datos (dict): El objeto JSON que deseas guardar.
    nombre_archivo (str): El nombre del archivo sin extensión, se añadirá ".json" automáticamente.
    """
    try:
        # Guarda el archivo con la extensión .json
        with open(f"{nombre_archivo}.json", "w") as archivo:
            json.dump(datos, archivo, indent=4)
        print(f"Archivo guardado exitosamente como {nombre_archivo}.json")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

            
def create_index_if_not_exists(client, index_name):
    """Creates an index with the specified name if it does not already exist.
    
    Args:
        index_name (str): The name of the index to create or check for existence.
    
    Returns:
        Index: Instance of the created or existing index.
    """

    # List the names of existing indexes
    existing_indexes = [index_info['name'] for index_info in client.list_indexes()]

    # Check if the specified index name is not in the list of existing indexes
    if index_name not in existing_indexes:
        # Create the index with the specified parameters
        client.create_index(
            name=index_name,  # Name of the index
            dimension=1_024,    # Dimension of the vectors
            metric='dotproduct',   # Similarity metric
            spec=ServerlessSpec(cloud='aws', region='us-east-1'), # Server configuration
        )

        # Wait until the index is ready to use
        while not client.describe_index(index_name).status['ready']:
            time.sleep(1)  # Pause to avoid overly frequent checks

        # Confirm that the index has been created
        print(f'Index {index_name} created')
    else:
        # Indicate that the index already exists
        print(f'The index {index_name} already exists')

    # Get an instance of the (created or existing) index and return it
    index = client.Index(index_name)
    return index


def upsert_vectors_in_batches(vectors, index, batch_size=100):
    """
    Upserts a list of vectors in batches to the provided index.

    This function splits the provided list of vectors into smaller batches and
    performs the upsert operation on each batch to avoid handling too much data at once.

    Args:
        vectors (list): A pre-generated list of vectors to upsert.
        index: An index instance with an upsert method to insert the vectors.
        batch_size (int): The number of vectors to include in each batch. Defaults to 100.
    
    Returns:
        None
    """
    # Process the list of vectors in batches
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        
        
def get_credentials(root_dir, SCOPES):
    """
    Retrieve or generate Google API credentials.

    Args:
        root_dir (Path): The root directory containing credentials and token files.
        SCOPES (list): The list of OAuth scopes required for the API access.

    Returns:
        Credentials: Authorized Google API credentials.
    """
    creds = None
    token_path = root_dir / 'token.json'

    # Check if a token file exists
    if os.path.exists(token_path):
        with open(token_path, 'r') as token:
            try:
                # Load token data and create credentials
                creds_info = json.load(token)
                creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
            except json.JSONDecodeError:
                # Handle invalid token JSON
                creds = None

    # Verify credentials and ensure the required scopes are included
    if creds and creds.valid and creds.has_scopes(SCOPES):
        return creds
    else:
        # Remove invalid token file if it exists
        if os.path.exists(token_path):
            os.remove(token_path)

        # Initiate OAuth flow to generate new credentials
        flow = InstalledAppFlow.from_client_secrets_file(
            root_dir / 'credentials.json', SCOPES
        )
        creds = flow.run_local_server(port=0)

        # Save the new credentials to the token file
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

        return creds
    

def create_usuarios_table(conn):
    """
    Create the 'usuarios' table in the provided SQLite database connection if it doesn't already exist.

    The table 'usuarios' will have the following columns:
      - id_usuario: an TEXT PRIMARY KEY that uniquely identifies each user.
      - nombre: a TEXT field for the user's name.
      - profesion: a TEXT field for the user's profession.
      - gmail: a TEXT field for the user's Gmail address.

    Parameters:
    conn (sqlite3.Connection): An open SQLite database connection.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id_usuario TEXT PRIMARY KEY,
            nombre TEXT,
            profesion TEXT,
            gmail TEXT
        )
    """)
    conn.commit()


def list_tables(conn):
    """
    List all tables in the provided SQLite database connection.
    
    Parameters:
    conn (sqlite3.Connection): An open SQLite database connection.
    
    Returns:
    list: A list of table names in the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def get_user_info(conn, user_id):
    """
    Retrieve a user's information from the 'usuarios' table based on the provided user ID.
    
    Parameters:
    conn (sqlite3.Connection): An open SQLite database connection.
    user_id (int): The id of the user to retrieve.
    
    Returns:
    dict or None: A dictionary with keys 'id_usuario', 'nombre', 'profesion', and 'gmail' if the user exists,
                  otherwise None.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id_usuario, nombre, profesion, gmail FROM usuarios WHERE id_usuario = ?", (user_id,))
    row = cursor.fetchone()
    
    if row:
        return {
            'nombre': row[1],
            'profesion': row[2],
            'gmail': row[3]
        }
    return None


def upsert_user(conn, user_id, user_data):
    """
    Insert a new user or update an existing user's information in the 'usuarios' table.

    Parameters:
    conn (sqlite3.Connection): An open SQLite database connection.
    user_id (int): The id of the user to update or insert.
    user_data (dict): A dictionary with keys among 'nombre', 'profesion', and 'gmail'. The values
                      are the corresponding field values for the user.
    """
    cursor = conn.cursor()
    
    # Check if the user exists.
    cursor.execute("SELECT id_usuario FROM usuarios WHERE id_usuario = ?", (user_id,))
    exists = cursor.fetchone() is not None
    
    if exists:
        # If there are fields to update, build and execute the UPDATE statement.
        if user_data:
            update_clause = ", ".join([f"{key} = ?" for key in user_data.keys()])
            values = list(user_data.values())
            values.append(user_id)
            sql = f"UPDATE usuarios SET {update_clause} WHERE id_usuario = ?"
            cursor.execute(sql, values)
    else:
        # Insert new record. Merge the user_id with the provided user_data.
        columns = ["id_usuario"] + list(user_data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        values = [user_id] + list(user_data.values())
        sql = f"INSERT INTO usuarios ({', '.join(columns)}) VALUES ({placeholders})"
        cursor.execute(sql, values)
    
    conn.commit()


def delete_user(conn, user_id):
    """
    Delete a user from the 'usuarios' table based on the provided user ID.

    Parameters:
    conn (sqlite3.Connection): An open SQLite database connection.
    user_id (int): The id of the user to delete.

    Returns:
    bool: True if a record was deleted, False otherwise.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM usuarios WHERE id_usuario = ?", (user_id,))
    conn.commit()
    return cursor.rowcount > 0