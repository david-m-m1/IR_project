import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing

PROJECT_ID = 'firm-catalyst-414509'
def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)

# Let's start with a small block size of 30 bytes just to test things out. 
BLOCK_SIZE = 1999998

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'), 
                                'wb', self._bucket) 
                          for i in itertools.count())
        self._f = next(self._file_gen)
           
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            f_name = str(self._base_dir / f_name)
            if f_name not in self._open_files:
                self._open_files[f_name] = _open(f_name, 'rb', self._bucket)
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)
  
    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False 

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # dict that stores for each document the len of the doc.
        self.document_len = Counter()
        # dict that stores for each document the title.
        self.doc_title_dict = Counter()
        # average document length
        self.avg_doc_len = 0
        self.doc_amount = 0
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)
        self.dictIndexTerm = Counter()

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        w2cnt = Counter(tokens)
        print(len(tokens))
        self.document_len[doc_id] = len(tokens)
        print(self.document_len[doc_id])
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        dictIndexTerm = Counter()
        counter = 0
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl: 
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
                dictIndexTerm[w] = counter
                counter += 1
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id,posting_locs


    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)

#     @staticmethod
#     def write_dict(data_dict, dict_name, bucket_name, chunk_size=100000):
#         """
#         Write a dictionary to pickle files in Google Cloud Storage (GCS) in chunks.
#
#         Parameters:
#
# data_dict: The dictionary to be serialized.
# bucket_name: The name of the Google Cloud Storage bucket.
# chunk_size: The number of entries to write per pickle file. Default is 100,000."""  # Initialize Google Cloud Storage client
#         client = storage.Client(PROJECT_ID)
#
#         # Get the bucket
#         bucket = client.get_bucket(bucket_name)
#
#         # Split the dictionary into chunks
#         chunks = [list(data_dict.items())[i:i + chunk_size] for i in range(0, len(data_dict), chunk_size)]
#
#         # Write each chunk to a separate pickle file in GCS
#         for i, chunk in enumerate(chunks):
#             blob = bucket.blob(f'{dict_name + ""}datachunk{i}.pickle')
#             with blob.open("wb") as f:
#                 pickle.dump(dict(chunk), f)
#
#     @staticmethod
#     def read_dict(dict_name, bucket_name):
#         """
#         Read a dictionary from pickle files in Google Cloud Storage (GCS) and combine them into a single dictionary.
#
#         Parameters:
#
# dict_name: name of the dictionary as was written
# bucket_name: The name of your Google Cloud Storage bucket.
#
#         Returns:
#
# A dictionary containing data from all pickle files."""  # Initialize Google Cloud Storage client
#         client = storage.Client(PROJECT_ID)
#
#         # Get the bucket
#         bucket = client.get_bucket(bucket_name)
#
#         data_dict = {}
#         # Iterate over blobs in the bucket
#         for blob in bucket.list_blobs():
#             if blob.name.endswith('.pickle') and blob.name.startswith(dict_name):
#                 data = blob.download_as_string()
#                 chunk_data = pickle.loads(data)
#                 data_dict.update(chunk_data)
#         return data_dict



    @staticmethod
    def write_dict(data_dict, dict_name, bucket_name, path, chunk_size=100000):
        """
        Write a dictionary to pickle files in Google Cloud Storage (GCS) in chunks.

        Parameters:
        - data_dict: The dictionary to be serialized.
        - dict_name: The name of the dictionary.
        - bucket_name: The name of the Google Cloud Storage bucket.
        - path: The path to write the files to.
        - chunk_size: The number of entries to write per pickle file. Default is 100,000.
        """
        # Initialize Google Cloud Storage client
        client = storage.Client(PROJECT_ID)

        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        # Split the dictionary into chunks
        chunks = [list(data_dict.items())[i:i + chunk_size] for i in range(0, len(data_dict), chunk_size)]

        # Write each chunk to a separate pickle file in GCS
        for i, chunk in enumerate(chunks):
            blob = bucket.blob(os.path.join(path, f'{dict_name}datachunk{i}.pickle'))
            with blob.open("wb") as f:
                pickle.dump(dict(chunk), f)

    @staticmethod
    def read_dict(dict_name, bucket_name, path):
        """
        Read a dictionary from pickle files in Google Cloud Storage (GCS) and combine them into a single dictionary.

        Parameters:
        - dict_name: The name of the dictionary.
        - bucket_name: The name of your Google Cloud Storage bucket.
        - path: The path to read the files from.

        Returns:
        A dictionary containing data from all pickle files.
        """
        # Initialize Google Cloud Storage client
        client = storage.Client(PROJECT_ID)

        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        data_dict = {}
        # Iterate over blobs in the bucket
        for blob in bucket.list_blobs():
            if blob.name.endswith('.pickle') and blob.name.startswith(os.path.join(path, dict_name)):
                data = blob.download_as_string()
                chunk_data = pickle.loads(data)
                data_dict.update(chunk_data)
        return data_dict
