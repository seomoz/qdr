
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

        

if __name__ == '__main__':
    unittest.main()

