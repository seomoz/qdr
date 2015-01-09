
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


if __name__ == '__main__':
    unittest.main()

