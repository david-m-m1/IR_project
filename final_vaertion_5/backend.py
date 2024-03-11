from collections import Counter
from google.cloud import storage
import pickle
from google.cloud import storage
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
import numpy as np
import heapq

#downloads
nltk.download('punkt')
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
stemmer = PorterStemmer()
bucket_name = '213047285'
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

path_to_dicts = "inverted_on_text_without_stem/dicts_folder"

#create inverted_on_text
def load_data_text():
    path_to_dicts = "inverted_on_text/dicts_folder"
    bucket_name = '213047285'
    file_path1 = "inverted_on_text_without_stem/inverted_text_index_v5.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path1)
    contents = blob.download_as_bytes()
    inverted = pickle.loads(contents)
    inverted.document_len = inverted.read_dict("doc_len", bucket_name, path_to_dicts)
    inverted.doc_title_dict = inverted.read_dict("doc_title", bucket_name, path_to_dicts)
    return inverted

#create inverted_on_title
def load_data_title(inverted):
    bucket_name = '213047285'
    file_path2 = "inverted_on_title_with_stem/inverted_title_index_v5.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path2)
    contents = blob.download_as_bytes()
    inverted_title1 = pickle.loads(contents)
    inverted_title1.doc_title_dict = inverted.doc_title_dict
    return inverted_title1

# load the data into memory
inverted_text = load_data_text()
inverted_title = load_data_title(inverted_text)
page_rank_dict = inverted_text.read_dict("doc_p", bucket_name, path_to_dicts)


def preprocess_query(query):
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]

    return filtered_tokens


def preprocess_query_stem(query):
    # Initialize Porter Stemmer
    stemmer = PorterStemmer()

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]

    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Perform additional preprocessing if needed

    return stemmed_tokens


def vectorize_query(query, inverted):
    # Preprocess the query
    preprocessed_query = preprocess_query(query)

    dict_tokens_unq = Counter(preprocessed_query)
    # Initialize a vector for the query
    query_vector = np.zeros(len(dict_tokens_unq))  # to do:create vector in size of the query
    counter = 0
    # Calculate TF-IDF for the query terms
    for term, freq in dict_tokens_unq.items():

        if inverted.posting_locs.get(term) is not None:
            # Calculate TF (term frequency) for the query term

            tf = freq / len(preprocessed_query)

            # Calculate TF-IDF score for the query term
            tf_idf = tf * 1

            # Assign TF-IDF score to the corresponding dimension in the query vector
            query_vector[counter] = tf_idf
            counter += 1
    return query_vector


def vectorize_documents(inverted, query_tokens_unq):
    # Initialize a dictionary to store document vectors
    document_vectors = {}

    counter = 0
    # Calculate TF-IDF for each document
    for term in query_tokens_unq:
        # if term not in inverted.df.keys()
        if inverted.df.get(term) is None:
            counter += 1
            continue
        # posting_list = inverted.read_a_posting_list(base_dir, term, bucket_name)
        posting_list = inverted.read_a_posting_list("", term, bucket_name)
        num_of_docs = len(inverted.document_len.items())
        df_of_term = inverted.df[term]
        idf = np.log(num_of_docs / df_of_term)  # Adding 1 to avoid division by zero
        for doc_id, tf in posting_list:
            tf = tf / (inverted.document_len[doc_id])
            # tf = tf / (inverted.document_len[doc_id])# returnnn

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


def BM25_similarity(inverted, dict_query_term_tf, number_to_return, k1, k2, b, is_text=False, value_factor=1):
    doc_amount = 6348910
    sum_bm_value = 0
    # return top results sorted.
    query_tokens_unq = list(dict_query_term_tf.keys())

    similarities = {}
    match_p = {}
    for term in query_tokens_unq:
        # if term not in inverted.df.keys()
        if inverted.posting_locs.get(term) is None:
            continue
        if inverted.df.get(term) is None:
            continue
        posting_list = inverted.read_a_posting_list("", term, bucket_name)
        # num_of_docs = len(inverted.document_len.items())
        num_of_docs = doc_amount
        df_of_term = inverted.df[term]
        idf = np.log((num_of_docs + 1) / df_of_term)  # Adding 1 to avoid division by zero
        posting_list_len = len(posting_list)
        for doc_id, tf in posting_list:
            if is_text:
                B = (1 - b) + (b * (inverted.document_len[doc_id] / inverted.avg_doc_len))
            else:
                B = (1 - b) + (b * (len(inverted.doc_title_dict[doc_id]) / inverted.avg_doc_len))
            query_tf = dict_query_term_tf[term]
            bm25_value_iteration = ((k1 + 1) * tf) / (B * k1 + tf) * idf * ((k2 + 1) * query_tf) / (k2 + query_tf)
            # If the document vector already exists, update it
            if similarities.get(doc_id) != None:
                old_factor = value_factor
                if is_text == False:
                    match_p[doc_id] = match_p[doc_id] + 1
                    if match_p[doc_id] / len(query_tokens_unq) == 1:
                        value_factor = (1 - value_factor) * 2
                    elif match_p[doc_id] / len(query_tokens_unq) > 0.65:
                        value_factor = (1 - value_factor) * 0.75
                similarities[doc_id] += bm25_value_iteration * value_factor
                sum_bm_value += bm25_value_iteration * value_factor
                value_factor = old_factor
            else:
                value_bm = bm25_value_iteration * value_factor
                # if ((tf < 4 and posting_list_len > 300000 and is_text == True) or (tf < 3 and posting_list_len > 150000 and is_text == True) or (tf < 2 and posting_list_len > 75000 and is_text == True)):# (tf < 3 and posting_list_len > 150000 and is_text == True) or
                #     continue
                old_factor = value_factor
                if is_text == False:
                    match_p[doc_id] = 1
                    if len(query_tokens_unq) == 1:
                        value_factor = (1 - value_factor) * 2
                similarities[doc_id] = bm25_value_iteration * value_factor
                sum_bm_value += bm25_value_iteration * value_factor
                value_factor = old_factor

    top_results = heapq.nlargest(number_to_return, similarities.items(), key=lambda x: x[1])

    # top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:number_to_return]
    return top_results


def search_text(query, inverted, k=50, use_cosine=False, value_factor_calc=0.85):
    """Search for documents based on a query using similarity."""


    query_vector = vectorize_query(query, inverted)
    processed_query = preprocess_query(query)
    list_tokens_unq = list(Counter(processed_query).keys())
    if use_cosine == True:
        vectorize_documents_res = vectorize_documents(inverted, list_tokens_unq)
        # Compute cosine similarity between the query vector and all document vectors
        similarities = {}
        for doc_id, doc_vector in vectorize_documents_res.items():
            similarities[doc_id] = cosine_similarity(query_vector, doc_vector)

        # Sort documents by similarity and return the top k results
        top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    # use MB25
    else:
        k1 = 5
        k2 = 2
        b = 0.5
        is_text = True
        top_results = BM25_similarity(inverted, Counter(processed_query), k, k1, k2, b, is_text, value_factor_calc)
    return top_results


def search_title(query, inverted_title, k=50, use_cosine=False, value_factor_calc=0.15):
    """Search for documents based on a query using similarity."""
    # Preprocess query (e.g., tokenize, remove stop words, etc.)
    # Vectorize the query

    query_vector = vectorize_query(query, inverted_title)
    processed_query = preprocess_query_stem(query)
    list_tokens_unq = list(Counter(processed_query).keys())
    if use_cosine == True:
        vectorize_documents_res = vectorize_documents(inverted_title, list_tokens_unq)
        # Compute cosine similarity between the query vector and all document vectors
        similarities = {}
        for doc_id, doc_vector in vectorize_documents_res.items():
            similarities[doc_id] = cosine_similarity(query_vector, doc_vector)

        # Sort documents by similarity and return the top k results
        top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    # use MB25
    else:
        k1 = 2
        k2 = 2
        b = 0.5
        is_text = False
        top_results = BM25_similarity(inverted_title, Counter(processed_query), k, k1, k2, b, False, value_factor_calc)

    return top_results
    # # Retrieve titles for the top results
    # results_with_titles = [(doc_id, inverted_index.dict_doc_title.get(doc_id)) for doc_id, _ in top_results]
    # #results_with_titles = [(doc_id, doc_titles[doc_id]) for doc_id, _ in top_results]
    # return results_with_titles


def search(query):
    top_results_text = search_text(query, inverted_text, 250, False, 0.8)
    top_results_title = search_title(query, inverted_title, 250, False, 0.2)

    # Initialize a Counter object to store the combined counts
    dict_of_ans = Counter()

    # Iterate over top_results_text and top_results_title to update the Counter
    for doc_id, value in top_results_text:
        dict_of_ans[doc_id] += value

    for doc_id, value in top_results_title:
        dict_of_ans[doc_id] += value

    for doc_id, value in dict_of_ans.items():
        page_rank_factor = 0.5
        if page_rank_dict.get(doc_id) != None:
            dict_of_ans[doc_id] += page_rank_dict[doc_id] * page_rank_factor
    top_results_final_not_sorted = dict_of_ans.items()
    top_results_final = sorted(top_results_final_not_sorted, key=lambda x: x[1], reverse=True)[:100]

    results_with_titles = [(str(doc_id), inverted_text.doc_title_dict.get(doc_id)) for doc_id, _ in top_results_final]
    return results_with_titles
