from corpus import Document, NPChunkCorpus
from hmm import HMM
from unittest import TestCase, main
from evaluator import compute_cm
from random import shuffle, seed
import sys


class HMMTest(TestCase):
    u"""Tests for the HMM sequence labeler."""

    def split_np_chunk_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        sentences = NPChunkCorpus('np_chunking_wsj', document_class=document_class)
        seed(hash("np_chunk"))
        shuffle(sentences)
        return (sentences[:8936], sentences[8936:])

    def test_np_chunk_baseline(self):
        """Test NP chunking with word and postag feature"""
        train, test = self.split_np_chunk_corpus(Document)
        classifier = HMM()
        classifier.train(train)
        test_result = compute_cm(classifier, test)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i>=.90 for i in f1), 'not all greater than 90.0%')

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

