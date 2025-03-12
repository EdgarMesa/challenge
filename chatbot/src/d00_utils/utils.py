import os
from langchain_core.documents import Document
import hashlib
import unicodedata
import time
from tqdm import tqdm

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

def dict_to_document(estructured_dict, origen):
    """
    Converts a structured dictionary into a list of Document objects to insert in the index.
    
    :param estructured_dict: Dictionary where keys are articles and values are texts.
    :param origen: Source of the content, which will be added to the metadata and text.
    :return: List of Document objects with associated metadata.
    """
    
    documents = [] 
    
    for articulo, texto in estructured_dict.items():
        text_split = texto.split('\n')
        
        libro = text_split[0].strip().split(':')[-1]
        titulo = text_split[1].strip().split(':')[-1]
        capitulo = text_split[2].strip().split(':')[-1]
        seccion = text_split[3].strip().split(':')[-1]

        # Create metadata for the document
        metadata = {
            'origen': origen,
            'libro': libro,
            'titulo': titulo,
            'capitulo': capitulo,
            'seccion': seccion,
            'articulo': articulo
        }
        
        texto = f'{origen}\n{texto}'

        document = Document(
            page_content=texto,
            metadata=metadata)
        
        documents.append(document)
    
    return documents

def strip_accents(text):
    """
    Remove accent characters from the given string.

    This function normalizes the string into its decomposed form (NFKD) and then
    filters out any combining characters (i.e., the accent marks).

    Args:
        text (str): The input string from which accents should be removed.

    Returns:
        str: A new string with the accents removed.
    """
    
    if text is None:
        return None
    
    # Normalize the text to NFKD form.
    normalized_text = unicodedata.normalize('NFKD', text)
    
    # Creades de UUID for each document
    stripped_text = ''.join(
        char for char in normalized_text if not unicodedata.combining(char)
    )
    return stripped_text

def metadata_to_uuid(docs):
    """
    Generate UUID-like strings from document metadata.
    Args:
        docs (list): A list of document objects. Each document is expected to have a
                     'metadata' attribute that is a dictionary.

    Returns:
        list: A list of UUID-like strings generated from the documents' metadata.
    """
    # Create a list of initial UUID strings for each document.
    uuids = [
        '#'.join(
            [doc.metadata[k] if not isinstance(doc.metadata[k], bool) else k for k in doc.metadata]
        )
        for doc in docs
    ]
    
    final_uuids = []  
    
    # Process each initial UUID string.
    for _id in uuids:
        parts = _id.split('#', maxsplit=1)
        # Compute the SHA-256 hash of the second part
        hash_part = hashlib.sha256(parts[-1].encode('utf-8')).hexdigest()
        # Combine the first part and the hash (separated by '#') to form the final UUID.
        final_uuids.append(f'{strip_accents(parts[0])}#{hash_part}')
    
    return final_uuids


def generate_sparse_vector_in_batches(documents, embedding_client, fitted_bm25, batch_size=64):
    """
    Generate a list of vectors that combine sparse and dense embeddings for a set of documents
    in batches.

    Args:
        documents (list): A list of documents, each with attributes `metadata` and `page_content`.
        embedding_client: An embedding client instance with an `embed_documents` method to generate dense embeddings.
        fitted_bm25: A BM25 model instance with an `encode_documents` method to generate sparse embeddings.
        batch_size (int, optional): Number of documents to process in each batch.

    Returns:
        list: A list of dictionaries, where each dictionary contains the following keys:
              'id' (str): A unique identifier for the document.
              'sparse_values' (array-like): Sparse embedding values for the document.
              'values' (array-like): Dense embedding values for the document.
              'metadata' (dict): Metadata associated with the document.
    """

    # Generate a unique UUID for each document
    uuids = metadata_to_uuid(documents)
    
    for doc in documents:
        doc.metadata['text'] = doc.page_content

    vectors = []

    # Process documents in batches
    for start_idx in tqdm(range(0, len(documents), batch_size)):
        end_idx = start_idx + batch_size
        batch_docs = documents[start_idx:end_idx]
        batch_uuids = uuids[start_idx:end_idx]

        # Extract metadata for the batch
        batch_metadatas = [doc.metadata for doc in batch_docs]

        # Extract page_content for the batch
        batch_contents = [doc.page_content for doc in batch_docs]
        
        # Generate dense embeddings for the batch
        dense_embeds = embedding_client.embed(
                model=os.getenv('EMBEDDING_MODEL'),
                inputs=[d.page_content for d in batch_docs],
                parameters={'dimension':1_024,'input_type': 'passage', 'truncate': 'END'}
            )

        # Generate sparse embeddings using the fitted BM25 model for the batch
        sparse_values = fitted_bm25.encode_documents(batch_contents)

        # Combine the ID, sparse embedding, dense embedding, and metadata for each document in the batch
        for _id, sparse, dense, metadata in zip(batch_uuids, sparse_values, dense_embeds, batch_metadatas):
            vectors.append({
                'id': _id,
                'sparse_values': sparse,
                'values': dense['values'],
                'metadata': metadata,
            })
            
            
        time.sleep(20)

    return vectors



import json

def parse_tool_call(tool_call):
    """
    Parses a tool call dictionary into a formatted string.
    The function returns a string that includes the tool name and the parsed arguments.
    
    Args:
        tool_call (dict): A dictionary containing keys 'name' and 'arguments'.
    
    Returns:
        str: A formatted string with the function name and arguments.
    """
    name = tool_call.get('name', '<no name>')
    arguments_str = tool_call.get('arguments', '{}')
    
    try:
        # Parse the JSON string in 'arguments'
        arguments = json.loads(arguments_str)
        # Pretty-print the parsed JSON with indent and ensuring proper unicode display.
        formatted_arguments = json.dumps(arguments, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        # In case parsing fails, use the raw string.
        formatted_arguments = arguments_str

    result = f'Tool Call: {name}\nArguments:\n{formatted_arguments}'
    return result


def parse_graph_output(outputs, debug=False):
    """
    Parse graph outputs to extract messages in a human-friendly format.

    Args:
        outputs (list): List of output nodes, each node is a dictionary containing messages.
        debug (bool): If True, include debug information (function calls and tool messages).

    Returns:
        list: A list of tuples, where each tuple consists of a message type and its content.
    """
    output_messages = []  

    for out in outputs:
        node_name = list(out.keys())[0]
        
        
        if node_name == '__interrupt__':
            for int in out[node_name]:
                output_messages.append(('ai', int.value['question']))
        else:
            if out[node_name]:
                # Extract the messages associated with this node.
                node_messages = out[node_name]['messages']
                
                # Ensure that node_messages is a list; if not, wrap it in a list for uniform processing.
                if not isinstance(node_messages, list):
                    node_messages = [node_messages]
                    
                for mess in node_messages:
                    # Check if the message is an instance of AIMessage.
                    if isinstance(mess, AIMessage):
                        # Determine if the AIMessage has additional keyword arguments (e.g., for function calls).
                        is_function_call = mess.additional_kwargs
                        if is_function_call:
                            # Extract the function call details from the additional keyword arguments.
                            function_call = mess.additional_kwargs['function_call']
                            if debug:
                                output_messages.append(('fn', str(parse_tool_call(function_call))))
                        else:
                            # Otherwise, treat it as a regular AI message.
                            output_messages.append(('ai', mess.content))
                            
                    elif isinstance(mess, ToolMessage) and debug:
                        # Append the tool message with a 'tool' tag and a prefixed label.
                        output_messages.append(('tool', f'Tool Message:\n{mess.content}'))
                
    return output_messages


def update_dict_preserve(dict_from, dict_to):
    """
    Update the keys of dict_to using values from dict_from. For any key present in both,
    if dict_to has a None value and dict_from has a non-None value, the value from dict_from
    will be used. Otherwise, dict_to's existing non-None value is preserved.
    
    Parameters:
    dict_from (dict): The dictionary providing new values.
    dict_to (dict): The dictionary to be updated.
    
    Returns:
    dict: The updated dict_to.
    """
    for key, value in dict_from.items():
        if key in dict_to:
            # Only update if the value in dict_to is None and the new value is not None.
            if dict_to[key] is None and value is not None:
                dict_to[key] = value
        else:
            # If key is not present in dict_to, add it.
            dict_to[key] = value
    return dict_to