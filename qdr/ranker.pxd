cimport cython

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport uint64_t

# wrappers for the C++ classes we'll use
cdef extern from "_ranker.cc":
    ctypedef unordered_map[string, pair[uint64_t, uint64_t] ] counts_t
    ctypedef vector[string] doc_t
    ctypedef struct scores_t:
        double tfidf
        double bm25
        double lm_jm
        double lm_dirichlet
        double lm_ad
    cdef cppclass QDR:
        QDR(counts_t& counts_in, uint64_t total_docs)
        scores_t score(doc_t& document, doc_t& query) except +
        vector[scores_t] score_batch(
            doc_t& document, vector[doc_t]& queries) except +
        double get_idf(string)

# only need to define C attributes and methods here
cdef class QueryDocumentRelevance:
    cdef QDR *_qdr_ptr

