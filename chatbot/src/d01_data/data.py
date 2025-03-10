import json
from pinecone import ServerlessSpec
import time
from tqdm import tqdm

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
        
        


