from collections import Counter

from google.cloud import storage
import pickle

# load_from_bucket
# get textIndex.pkl from bucket
from google.cloud import storage
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
bucket_name = 'bucket-mr-project-ir-david'
file_path = "index_text_on_jason/inverted_index_small.pkl"
def load_data_text():
    bucket_name = 'bucket-mr-project-ir-david'
    file_path = "index_text_on_jason/inverted_index_small.pkl"
    # load_from_bucket
    # get textIndex.pkl from bucket
    path_to_dicts = "inverted_on_text/dicts_folder"
    from google.cloud import storage
    import pickle
    bucket_name = 'bucket-mr-project-ir-david'
    file_path = "inverted_on_text_without_stem/inverted_text_index_v5.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    inverted = pickle.loads(contents)
    inverted.document_len = inverted.read_dict("doc_len", bucket_name, path_to_dicts)

    # num_dictionaries = 11
    # doc_title_dictionaries = []
    # for i in range(num_dictionaries):
    #     dict_i = inverted.read_dict(f'doc_title{i}',bucket_name,path_to_dicts)
    #     doc_title_dictionaries.append(dict_i)

    inverted.doc_title_dict = inverted.read_dict("doc_title", bucket_name, path_to_dicts)
    return inverted
def load_data_title():
    bucket_name = 'bucket-mr-project-ir-david'
    file_path = "index_text_on_jason/inverted_index_small.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    inverted = pickle.loads(contents)

    inverted.document_len = inverted.read_dict("doc_len",bucket_name)
    inverted.doc_title_dict = inverted.read_dict("doc_title",bucket_name)
    return inverted

def preprocess_query(query):
    # Download NLTK resources if not already downloaded //make sure documents are preprocess in the same way
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    nltk.download('punkt')
    nltk.download('stopwords')
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    # Initialize Porter Stemmer
    stemmer = PorterStemmer()

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]

    # Stemming
    stemmed_tokens = [token for token in filtered_tokens]
    #stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Perform additional preprocessing if needed

    return stemmed_tokens

def vectorize_query(query, inverted_index):
    # Preprocess the query
    preprocessed_query = preprocess_query(query)

    dict_tokens_unq = Counter(preprocessed_query)
    # Initialize a vector for the query
    query_vector = np.zeros(len(dict_tokens_unq))# to do:create vector in size of the query
    counter = 0
    # Calculate TF-IDF for the query terms
    for term, freq in dict_tokens_unq.items():

        if inverted_index.posting_locs.get(term) is not None:
            # Calculate TF (term frequency) for the query term

            tf = freq/len(preprocessed_query)

            # Calculate TF-IDF score for the query term
            tf_idf = tf * 1

            # Assign TF-IDF score to the corresponding dimension in the query vector
            query_vector[counter] = tf_idf
            counter += 1
    return query_vector


def vectorize_documents(inverted,query_tokens_unq):
    # Initialize a dictionary to store document vectors
    document_vectors = {}



    counter = 0
    # Calculate TF-IDF for each document
    for term in query_tokens_unq:
        #if term not in inverted.df.keys()
        if inverted.df.get(term) is None:
          counter += 1
          continue
        # posting_list = inverted.read_a_posting_list(base_dir, term, bucket_name)
        posting_list = inverted.read_a_posting_list("",term, bucket_name)
        num_of_docs = len(inverted.document_len.items())
        print(str(num_of_docs)+"num_of_docs")
        df_of_term = inverted.df[term]
        idf = np.log(num_of_docs/df_of_term)  # Adding 1 to avoid division by zero
        print(idf)
        for doc_id, tf in posting_list:
            print(tf)
            tf = tf / (inverted.document_len[doc_id])
            print(tf)
            #tf = tf / (inverted.document_len[doc_id])# returnnn

            # Calculate TF-IDF score for the term in the document
            tf_idf = tf * idf

            # If the document vector already exists, update it
            if doc_id in document_vectors:
                document_vectors[doc_id][counter] = tf_idf
            # Otherwise, create a new document vector
            else:
                document_vector = np.zeros(len(query_tokens_unq))
                document_vectors[doc_id] = document_vector
                document_vectors[doc_id][counter] = tf_idf
        counter += 1
    return document_vectors
def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)
def MB25_similarity(inverted, dict_query_term_tf,number_to_return,k1,k2, b):
    #return top results sorted.
    query_tokens_unq = list(dict_query_term_tf.keys())
    similarities = {}
    for term in query_tokens_unq:
        #if term not in inverted.df.keys()
        if inverted.df.get(term) is None:
          continue
        posting_list = inverted.read_a_posting_list("",term, bucket_name)
        num_of_docs = len(inverted.document_len.items())
        df_of_term = inverted.df[term]
        idf = np.log(num_of_docs+1/df_of_term)  # Adding 1 to avoid division by zero
        print(idf)
        for doc_id, tf in posting_list:
            B = (1 - b )+ (b*(inverted.document_len[doc_id]/500))
            #B = (1 - b )+ (b*(inverted.document_len[doc_id]/inverted.avg_doc_len))
            query_tf =dict_query_term_tf[term]
            bm25_value_iteration = ((k1+1)*tf)/(B*k1+tf)*idf*((k2+1)*query_tf)/(k2+query_tf)
            # If the document vector already exists, update it
            if similarities.get(doc_id) != None:
                similarities[doc_id] += bm25_value_iteration
            else:
                similarities[doc_id] = bm25_value_iteration
    top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:number_to_return]
    return top_results

def search(query, inverted, k=10, use_cosine= True):
    """Search for documents based on a query using cosine similarity."""
    # Preprocess query (e.g., tokenize, remove stop words, etc.)
    # Vectorize the query

    query_vector = vectorize_query(query,inverted)
    processed_query = preprocess_query(query)
    list_tokens_unq = list(Counter(processed_query).keys())
    if use_cosine == True:
      vectorize_documents_res = vectorize_documents(inverted,list_tokens_unq)
      # Compute cosine similarity between the query vector and all document vectors
      similarities = {}
      for doc_id, doc_vector in vectorize_documents_res.items():
          similarities[doc_id] = cosine_similarity(query_vector, doc_vector)

      # Sort documents by similarity and return the top k results
      top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    #use MB25
    else:
      top_results = MB25_similarity(inverted,Counter(processed_query),k,k1=2,k2=2,b=0.5)

    # # Retrieve titles for the top results
    results_with_titles = [(str(doc_id), inverted.doc_title_dict.get(doc_id)) for doc_id, _ in top_results]
    # #results_with_titles = [(doc_id, doc_titles[doc_id]) for doc_id, _ in top_results]
    return results_with_titles

