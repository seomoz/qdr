qdr
===

[![Build Status](https://api.travis-ci.org/seomoz/qdr.png)](https://api.travis-ci.org/seomoz/qdr.png)

Query-Document Relevance ranking functions

This repository implements a few query-document similarity functions,
commonly used in information retrieval applications.  It supports:

* TF-IDF
* Okapi BM25
* Language Model

This implementation includes pure Python code for iteratively training
models from a large corpus, and a C++ implementation of the scoring
functions with Cython wrappers for fast evaluation.

Each of these ranking functions has a few "magic" constants.  Currently
these are hard coded to values recommend in the literature, but if the
need arises can be configurable.  Relevant references:

* For TF-IDF see [http://scholar.google.com/scholar?hl=en&as_sdt=0,48&q=salton+and+buckley+%22Term-weighting+approaches+in+automatic+text+retrieval%22+Information+Processing+%26+Management,+vol+24,+1988](Salton and Buckley, "Term-weighting approaches in automatic text retrieval") ("best fully weighted system tfc * nfx" (Table 2, first line))





