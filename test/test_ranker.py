
import unittest
import numpy as np

from qdr import ranker

# some test data
from common import *


class TestRanker(unittest.TestCase):
    def _get_qd(self):
        return ranker.QueryDocumentRelevance(corpus_unigrams, corpus_ndocs)

    def test_idf(self):
        qd = self._get_qd()
        self.assertAlmostEqual(qd.get_idf('deep'), np.log(corpus_ndocs / 1.0))
        self.assertAlmostEqual(qd.get_idf('the'), np.log(corpus_ndocs / 3.0))
        self.assertAlmostEqual(qd.get_idf('not_in_corpus'),
                               np.log(corpus_ndocs / 1.0))

    def test_empty(self):
        '''
        Empty queries or documents raise an exception
        '''
        qd = self._get_qd()
        self.assertRaises(ValueError, qd.score, [], query)
        self.assertRaises(ValueError, qd.score, document, [])
        self.assertRaises(ValueError, qd.score, [], [])
        self.assertRaises(ValueError, qd.score_batch, [], [query])
        self.assertRaises(ValueError, qd.score_batch, document, [query, []])
        # but empty queries in score_batch just return an empty list
        self.assertEqual(qd.score_batch(document, []), [])

    def test_tfidf(self):
        qd = self._get_qd()
        computed_score = qd.score(document, query)['tfidf']

        max_query_tf = 2.0
        query_vector = np.array([
                     (0.5 + 0.5 / max_query_tf) * qd.get_idf("buy"),
                     (0.5 + 0.5 / max_query_tf) * qd.get_idf("snow"),
                     (0.5 + 0.5 * 2.0 / max_query_tf) * qd.get_idf("shovel")])

        doc_vector = np.array([0.0,
                      2.0 * qd.get_idf("snow"),
                      1.0 * qd.get_idf("shovel")])
        doc_length = np.sqrt(np.sum(np.array([
                        1.0 * qd.get_idf("the"),
                        1.0 * qd.get_idf("store"),
                        1.0 * qd.get_idf("sells"),
                        2.0 * qd.get_idf("snow"),
                        1.0 * qd.get_idf("shovel")]) ** 2))

        expected_score = np.sum(query_vector * doc_vector) / doc_length
        self.assertAlmostEqual(computed_score, expected_score)

    def test_bm25(self):
        qd = self._get_qd()
        computed_score = qd.score(document, query)['bm25']

        # SUM_{t in query} log(N / df[t]) * (k1 + 1) * tf[td] /
        #                  (k1 * ((1 - b) + b * (Ld / Lave)) + tf[td])

        k1 = 1.6
        b = 0.75
        Lave = sum([len(ele) for ele in corpus]
                 + [len(ele) for ele in corpus_update]) / float(corpus_ndocs)

        score_buy = 0.0 # not in document
        score_snow = np.log(float(corpus_ndocs) / corpus_unigrams['snow'][1]) \
            * (k1 + 1.0) * 2.0 / (k1 * ((1.0 - b) + b * (6.0 / Lave)) + 2.0)
        score_shovel = \
            np.log(float(corpus_ndocs) / corpus_unigrams['shovel'][1]) \
            * (k1 + 1.0) * 1.0 / (k1 * ((1.0 - b) + b * (6.0 / Lave)) + 1.0)
        actual_score = score_buy + score_snow + 2.0 * score_shovel

        self.assertAlmostEqual(computed_score, actual_score)

    def test_lm(self):
        qd = self._get_qd()
        computed_scores = qd.score(document, query)

        lam = 0.1
        mu = 2000.0
        delta = 0.7

        jm = 0.0
        dirichlet = 0.0
        ad = 0.0
        sum_w_cwd_doc = float(len(document))
        nwords_corpus = sum(v[0] for v in corpus_unigrams.itervalues())
        n2p1 = len(corpus_unigrams) + nwords_corpus + 1
        for word in query:
            try:
                word_count_corpus = corpus_unigrams[word][0]
            except KeyError:
                word_count_corpus = 0
            corpus_prob = (word_count_corpus + 1.0) / n2p1

            cwd = 0
            for doc_word in document:
                if doc_word == word:
                    cwd += 1

            if cwd == 0:
                # not in document
                jm += np.log(lam * corpus_prob)
                dirichlet += np.log(mu / (sum_w_cwd_doc + mu) * corpus_prob)
                ad += np.log(
                    delta * len(set(document)) / sum_w_cwd_doc * corpus_prob)
            else:
                jm += np.log(
                        (1.0 - lam) * cwd / sum_w_cwd_doc + lam * corpus_prob)
                dirichlet += np.log(
                        (cwd + mu * corpus_prob) / (sum_w_cwd_doc + mu))
                ad += np.log(
                   max(cwd - delta, 0.0) / sum_w_cwd_doc +
                   delta * len(set(document)) / sum_w_cwd_doc * corpus_prob)

        self.assertAlmostEqual(computed_scores['lm_jm'], jm)
        self.assertAlmostEqual(computed_scores['lm_dirichlet'], dirichlet)
        self.assertAlmostEqual(computed_scores['lm_ad'], ad)

    def test_load_from_file(self):
        import os
        from tempfile import mkstemp
        from qdr.trainer import write_model

        t = mkstemp()
        write_model(corpus_ndocs, corpus_unigrams, t[1])
        qd = ranker.QueryDocumentRelevance.load_from_file(t[1])

        # we'll just check that one of the word counts is correct
        self.assertAlmostEqual(qd.get_idf('the'), np.log(corpus_ndocs / 3.0))

        os.remove(t[1])

    def test_score_batch(self):
        qd = self._get_qd()
        queries = [query, ['buy', 'shovel']]
        scores = qd.score_batch(document, queries)
        # we'll assume that the score single works...
        self.assertEqual(len(scores), 2)


if __name__ == '__main__':
    unittest.main()

