qdr
===

[![Build Status](https://api.travis-ci.org/seomoz/qdr.png)](https://travis-ci.org/seomoz/qdr)

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

* For TF-IDF see [Salton and Buckley, "Term-weighting approaches in automatic text retrieval"](http://scholar.google.com/scholar?hl=en&as_sdt=0,48&q=salton+and+buckley+%22Term-weighting+approaches+in+automatic+text+retrieval%22+Information+Processing+%26+Management,+vol+24,+1988) ("best fully weighted system tfc * nfx" (Table 2, first line))
* For Okapi BM25, see ["An Introduction to Information Retrieval" by Manning et al.](http://nlp.stanford.edu/IR-book/) (Section 11.4.3 (page 233), eqn 11.32).
* For the Language Model approach, see [Zhai and Lafferty "A Study of Smoothing
Methods for Language Models Applied to Ad Hoc Information Retrieval"](http://scholar.google.com/scholar?q=Zhai+and+Lafferty+"A+Study+of+Smoothing+Methods+for+Language+Models+Applied+to+Ad+Hoc+Information+Retrieval")

Usage
=====

All tokenization and word normalization is handled client side, and all methods
that accept queries or documents assume they are lists of byte strings,
not unicode.

There are two separate steps to using the ranking functions: training
and scoring.

## Training

The `Trainer` class supports incremental training from a large corpus,
combining separately trained models for map-reduce type data flows,
pruning of infrequent tokens from large models and serialization.  Typical
usage:

```python
from qdr import Trainer

# load corpus -- it's an iterable of documents, each document is a
# list of byte strings
model = Trainer()
model.train(corpus)

# the train method adds documents incrementally so it can be updated with
# additional documents by calling train again
model.train(another_corpus)

# write to a file
model.serialize_to_file(outputfile)
```

For map-reduce type work, the method `update_counts_from_trained` will
merge the contents of two `Trainer` instances:

```python
# map step -- typically this is parallelized
for k, corpus in enumerate(corpus_chunks):
    model = Trainer()
    model.train(corpus)
    model.serialize_to_file("file%s.gz" % k)

# reduce step
model = Trainer.load_from_file("file0.gz")
for k in xrange(1, len(corpus_chunks)):
    model2 = Trainer.load_from_file("file%s.gz" % k)
    model.update_counts_from_trained(model2)

# prune the final model if needed
model.prune(min_word_count, min_doc_count)
```

## Scoring

Typical usage:

```python
from qdr import QueryDocumentRelevance

scorer = QueryDocumentRelevance.load_from_file('trained_model.gz')
# document, query are lists of byte strings
relevance_scores = scorer.score(document, query)
```

For scoring batches of queries against a single document, the `score_batch`
method is more efficient then calling `score` repeatedly:
```
# queries is a list of queries, each query is a list of tokens:
relevance_scores = scorer.score(document, queries)
```

# Installing

```
sudo pip install -r requirements.txt
sudo make install
```

# Contributing
Contributions welcome!  Fork, commit, then open a pull request.


