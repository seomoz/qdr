
import os
import unittest

from tempfile import mkstemp 

from qdr import trainer


# some test data
corpus = ["he went down to the store".split(),
        "he needed a shovel from the store to shovel the snow".split()]
corpus_update = ["the snow was five feet deep".split()]

corpus_unigrams = {
 'a': [1, 1],
 'deep': [1, 1],
 'down': [1, 1],
 'feet': [1, 1],
 'five': [1, 1],
 'from': [1, 1],
 'he': [2, 2],
 'needed': [1, 1],
 'shovel': [2, 1],
 'snow': [2, 2],
 'store': [2, 2],
 'the': [4, 3],
 'to': [2, 2],
 'was': [1, 1],
 'went': [1, 1],
}
corpus_ndocs = 3


class TestTrainer(unittest.TestCase):
    def _get_qd(self):
        qd = trainer.Trainer()
        qd.train(corpus)
        qd.train(corpus_update)
        return qd

    def test_train(self):
        qd = self._get_qd()
        self.assertEqual(qd._counts, corpus_unigrams)
        self.assertEqual(qd._total_docs, corpus_ndocs)

    def test_update_from_trained(self):
        qd = trainer.Trainer()
        qd.train(corpus)
        qd2 = trainer.Trainer()
        qd2.train(corpus_update)
        qd.update_counts_from_trained(qd2)

        self.assertEqual(qd._counts, corpus_unigrams)
        self.assertEqual(qd._total_docs, corpus_ndocs)

    def test_serialize(self):
        '''
        We should be able to write out the model then read it back in
        '''
        qd = self._get_qd()
        t = mkstemp()
        qd.serialize_to_file(t[1])

        # load from file and check it
        qd2 = trainer.Trainer.load_from_file(t[1])
        self.assertEqual(qd2._counts, corpus_unigrams)
        self.assertEqual(qd2._total_docs, corpus_ndocs)

        os.unlink(t[1])

    def test_prune(self):
        qd = self._get_qd()
        qd.prune(2, 0)

        self.assertEqual(qd._counts,
            {'he': [2, 2], 'shovel': [2, 1], 'snow': [2, 2], 'store': [2, 2],
             'the': [4, 3], 'to': [2, 2]})

        qd.prune(2, 3)
        self.assertEqual(qd._counts, {'the': [4, 3]})

if __name__ == '__main__':
    unittest.main()

