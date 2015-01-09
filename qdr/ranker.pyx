
# c imports
cimport cython
from ranker cimport *

cdef class QueryDocumentRelevance:
    def __cinit__(self, counts, total_docs):
        '''
        Load the model and construct the C++ class

        counts: the token -> (word count, document count) map from the corpus
        total_docs: total documents in the corpus
        '''
        self._qdr_ptr = new QDR(counts, total_docs)

    def __dealloc__(self):
        del self._qdr_ptr

    def score(self, document, query):
        '''
        Compute the query-document relevance scores

        document and query are tokenized lists of words
        '''
        # cython will handle the conversion for us...
        return self._qdr_ptr.score(document, query)

    def get_idf(self, word):
        return self._qdr_ptr.get_idf(word)

