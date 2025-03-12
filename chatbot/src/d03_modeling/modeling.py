import os 

from typing import List
from langchain_core.tools import tool
from pinecone import Pinecone

from pathlib import Path

from pinecone_text.sparse import BM25Encoder
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64

from src.d01_data.data import create_index_if_not_exists, get_credentials


root_dir = Path(os.getcwd()).parent.parent

output_path = root_dir / 'data' / '04_model_output'



SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
]

creds = get_credentials(root_dir=root_dir, SCOPES=SCOPES)
service = build('gmail', 'v1', credentials=creds)

pc = Pinecone()
pc_index = create_index_if_not_exists(client=pc, index_name='chatbot-leyes')

fitted_bm25 = BM25Encoder(language='spanish')
fitted_bm25.load(output_path / 'bm25_values.json')


def process_query(query):
    """
    Processes the query and extracts text from the metadata of each match.

    Args:
        query (dict): A dictionary containing a 'matches' key. Each match is 
                      expected to have a 'metadata' key with a 'text' value.

    Returns:
        str: A single string where the extracted text from all matches is joined 
             by two newline characters.
    """
    result_str = [] 
    
    for match in query['matches']: 
        result_str.append(match['metadata']['text'])  
    
    return '\n\n'.join(result_str)


def hybrid_scale(dense, sparse, alpha: float):
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError('Alpha must be between 0 and 1')
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


def hybrid_query(query, top_k, alpha, filter={}):
    # convert the query into a sparse vector
    sparse_vec = fitted_bm25.encode_documents([query])[0]
    
    dense_vec = pc.inference.embed(
                    model=os.getenv('EMBEDDING_MODEL'),
                    inputs=[query],
                    parameters={'dimension':1_024,'input_type': 'query', 'truncate': 'END'}
                )

    dense_vec = dense_vec[0]['values']

    # scale alpha with hybrid_scale
    dense_vec, sparse_vec = hybrid_scale(
    dense_vec, sparse_vec, alpha)
    
    # query pinecone with the query parameters
    result = pc_index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        filter=filter,
        top_k=top_k,
        include_metadata=True
    )
    # return search results as json
    return result


def rerank_document(query, docs, top_n=10, threshold=0):
    """
    Reranks documents using the specified reranking model and filters results
    based on a given threshold score.

    Args:
        query (str): The user's query.
        docs (list): List of documents to be reranked.
        threshold (float): Minimum score threshold for filtering results.

    Returns:
        list: Filtered list of documents that meet the threshold score.
    """
    
    # Perform reranking using the specified model
    rerank_docs = pc.inference.rerank(
        model=os.getenv('RERANK_MODEL'),
        query=query,
        documents=docs,
        top_n=top_n,
        return_documents=True,
            parameters={
                'truncate': 'END'
            }
    )

    # Filter documents based on the threshold
    filtered_docs = [
        doc for doc in rerank_docs.data
        if doc['score'] >= threshold
    ]

    return filtered_docs


def format_rerank_results(results):
    """
    Formats reranked results into a readable text format.

    Args:
        results (list): List of filtered rerank results.

    Returns:
        str: Formatted string with top result details.
    """
    formatted_texts = []

    for i, result in enumerate(results):
        formatted_text = (
            f'ranking: {i + 1}\n'
            f'puntuacion: {round(result["score"], 2)}\n'
            f'text: {result["document"]["text"]}\n'
            '---------------------------------------------'
        )
        formatted_texts.append(formatted_text)

    # Puts the best results at the end of the query
    return formatted_texts[::-1]



@tool
def consult_legal_database(query: str, alpha: float, filter: dict) -> list[str]:
    """
    Consults the legal vector database to retrieve legal documents from the Spanish legal corpus.

    This function employs a hybrid search strategy that combines both keyword-based and semantic search.
    The 'alpha' parameter controls the balance between the two:
      - alpha = 0: Use only keyword-based search.
      - alpha = 1: Use only semantic search.
    Intermediate values blend both approaches. This allows the model to adjust search behavior based on 
    the complexity or specificity of the user's query.

    Additionally, the function applies a metadata filter on the 'origen' field using the $in operator.
    This filter restricts the search to a specific legal document source, such as:
      - 'Código Civil'
      - 'Código Penal'
      - 'Constitucion Española'
    The filter dict should be structured as:
        {'origen': {'$in': [desired_sources]}}
    where [desired_sources] is one of the allowed legal texts, a combination of them or all of them.
    
    # Examples of a filter for 'Código Civil':
    # filter = {'origen': {'$in': ['Código Civil']}}
    # filter = {'origen': {'$in': ['Código Civil', 'Código Penal']}}

    Args:
        query (str): A user-provided query in natural language seeking legal information.
        alpha (float): A tuning parameter between 0 and 1 that balances keyword and semantic search.
        filter (dict): A metadata filter for the 'origen' field. It must use the $in operator and contain 
                       one of the following values: 'Código Civil', 'Código Penal', or 'Constitucion Española'.

    Returns:
        list[str]: A list of legal documents (as strings) from the database, sorted by relevance.
    """

    results = hybrid_query(
        query=query, 
        filter=filter,  
        top_k=40,  
        alpha=alpha
    )
    
    # Extract relevant text content from the query results
    results = [{
        'id': str(i),               
        'text': x['metadata']['text'] 
    } for i, x in enumerate(results['matches'])]

    # Rerank the filtered documents using a reranking model
    rerank_result = rerank_document(query=query, docs=results, top_n=8, threshold=0.5)

    if len(rerank_result) > 0:
        # Format the reranked results for improved readability
        rerank_result = format_rerank_results(rerank_result)
        rerank_result = '\n'.join(rerank_result)
        rerank_result = f'Información de la base de datos sobre la query del usuario:\n{rerank_result}'
    else:
        rerank_result = 'No hay información relevante a la query del usuario en la base de datos'

    return rerank_result


def _send_email(to: str, subject: str, message_text: str) -> str:

    """
    Sends an email using the provided service.
    
    Args:
        to (str): The recipient's email address.
        subject (str): The subject of the email.
        message_text (str): The plain text content of the email body.

    Returns:
        str. Returns a string saying that the email has been sent
    """
    #Quiero que me mandes todas esa informacion sobre langraph al correo edgarmp@metiora.com con el asunto informacion de langGraph
    #user_id (str, optional): The sender's user ID. Defaults to 'me'.
    # Create a multipart MIME message to hold email components
    message = MIMEMultipart()
    
    # Set the recipient email address
    message['to'] = to
    
    # Set the email subject
    message['subject'] = subject
    
    # Attach the plain text message to the email
    message.attach(MIMEText(message_text, 'plain'))
    
    # Encode the message as base64 for transmission
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    # Create the body of the message with the raw encoded content
    message_body = {'raw': raw}
    
    # Use the Gmail API to send the email via the authenticated user
    sent_message = (
        service.users().messages().send(userId='me', body=message_body).execute()
    )
    
    return sent_message

@tool
def send_email_f(to: str, subject: str, message_text: str) -> str:
    """
    Sends an email with the given subject and message text to the specified email address.

    This function is intended to be used when a user requests to send information via email.
    It requires that the user provides all necessary details:
      - A valid recipient email address (if not already available in the database, the email must be explicitly provided).
      - A subject for the email.
      - The message content to be sent.

    If any of these parameters (email address, subject, or message text) are missing, notify the user,
    to supply the missing information before attempting to send the email.

    Parameters:
        to (str): The recipient's email address.
        subject (str): The subject of the email.
        message_text (str): The body text of the email message.

    Returns:
        str: A confirmation message with the metadata of the email sent'

    Example:
            send_email(to="user@example.com", subject="Meeting Reminder", message_text="Don't forget our meeting at 10 AM.")
    """
    
    try:
        _send_email(to=to, subject=subject, message_text=message_text)
        return f'El correo con asunto "{subject}" se ha enviado correctamente a "{to}"'
    except Exception as e:
        return f'No se ha podido enviar el correo a {to}'