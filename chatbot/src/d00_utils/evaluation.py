import os

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


def hybrid_query(pc_index, fitted_bm25, embedding_model, query, top_k, alpha, filter={}):
    # convert the query into a sparse vector
    sparse_vec = fitted_bm25.encode_documents([query])[0]
    
    dense_vec = embedding_model.inference.embed(
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