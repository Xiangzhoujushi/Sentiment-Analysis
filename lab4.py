import gensim
import numpy as np
import os
import pandas as pd
import tensorflow as tf


# from lab4 import *; sdl = SentimentDeepLearner()
class SentimentDeepLearner(object):

    def __init__(self, word2vec_path='D:/Temp/GoogleNews-vectors-negative300.bin'):
        self.word2vec_path = word2vec_path
        self.base_folder = os.path.realpath(os.getcwd())
        self.class_labels = ['strongly negative', 'negative', 'neutral', 'positive', 'strongly positive']
        self.max_words = 56  # maximum number of words in a sentence (from analysis of SOStr.txt), 19.2 average
        self.embedding_size = 300  # from Google word2vec
        self.empty_embedding = np.zeros([1, self.embedding_size])

    def _build_labeled_sentences(self):
        """
        Create the dataset which maps sentence words to sentiment labels.

        @use
            Data Source: https://nlp.stanford.edu/sentiment/
            Should only need to be
        """

        # 1. Load the sentence partitions (training, test, dev)
        #   "datasetSplit.txt" (header): sentence_index,splitset_label
        partitions = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/datasetSplit.txt'), sep=',', index_col='sentence_index')

        # 2. Load the full sentences & join to partitions
        #   "datasetSentences.txt" (header): sentence_index[TAB]sentence
        sentences = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/datasetSentences.txt'), sep='\t', index_col='sentence_index')
        sentence_partitions = partitions.join(sentences)

        # 3. Load phrases & match to sentences
        #   "dictionary.txt" (no header):  <phrase> | <phrase_id>
        phrases = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/dictionary.txt'), sep='|', header=None)
        phrases.columns = ['phrase', 'phrase_id']  # no header in data file
        phrases = phrases.set_index('phrase_id')  # for joining to sentences

        # 4. Load phrase sentiments & match to phrases
        #   "sentiment_labels.txt" (header): phrase ids | sentiment values
        sentiments = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/sentiment_labels.txt'), sep='|', index_col='phrase ids')
        sentiments.columns = ['sentiment']  # has a header, but relabel
        phrase_sentiments = phrases.join(sentiments)

        # 5. Join phrases to sentences (sentences are subset of phrases)
        #   output: [splitset_label, sentence, phrase, sentiment]
        data = pd.merge(sentence_partitions, phrase_sentiments, left_on='sentence', right_on='phrase')
        data = data.drop(columns=['phrase'])  # remove redundant column

        # 6. Convert continuous sentiment to class label
        #   REF: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        data['sentiment_label'] = pd.cut(
            data['sentiment'],
            include_lowest=True,
            right=False,
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=self.class_labels
        )

        # 7. Load the sentence word parse as a single column & join to data
        #   ASSUMPTION: sentences are not indexed, assume same order as datasetSentences
        #   "SOStr.txt" (no header):  <word>|<word>|...
        words = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/SOStr.txt'), sep=' ', header=None)
        words.columns = ['wordstring']  # single column
        words['wordstring'] = words.apply(self._helper_standard_size, axis=1)  # regularize sentence length
        data = data.join(words)

        # 8. Drop "dev" labeled rows (this saves time on embedding)
        #   reduce rows from 11,855 to 10,243
        data = data.drop(data[data['splitset_label'] == 3].index)

        # 9. Save results for future use
        #   [splitset_label, sentiment, sentiment_label, wordstring]
        data = data.drop(columns=['sentence'])
        data.to_csv(os.path.join(self.base_folder, 'sentimentData/labeled_sentences.txt'), header=True, index=False, sep='\t')
        return data

    @staticmethod
    def _build_word_matrix(labeled_sentences):
        """
        Build a 2-D array with num_sentences x max_words.

        :param labeled_sentences: (pd.DataFrame)
            Output from self._build_labeled_sentences

        :return word_matrix, sentence_labels: (np.array, np.array)
        """
        data = []
        for irow, row in labeled_sentences.iterrows():
            data.append(row['wordstring'].split('|'))
        word_matrix = np.array(data)
        sentence_labels = np.array(labeled_sentences['sentiment_label'])
        return word_matrix, sentence_labels

    def _get_batch(self, word2vec, word_matrix, sentence_labels, batch_size):
        """
        Get a data for tensorflow by adding a new dimension (word embeddings) to sampling of word_matrix

        :param word2vec: (gensim.models.KeyedVectors)
            Pre-trained word embeddings.
        :param word_matrix: (np.array)
            The words in each sentence (sentences x max_words)
        :param sentence_labels: (np.array)
            The class label (sentences x 1)
        :param batch_size: (int)
            Number of random samples to use in batch

        :return batch, labels: (np.array, np.array)
            batch.shape = (batch_size, self.max_words, self.embedding_size)
            labels.shape = (1, batch_size)
        """
        # TODO: modify randomly sample sentences (instead of adjacent blocks)
        # see Helper Functions @ https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
        # TODO: return labels as well
        batch = np.zeros([len(word_matrix), self.max_words, self.embedding_size])  # len(word_matrix) == batch_size
        for isentence, sentence in enumerate(word_matrix):
            for iword, word in enumerate(sentence):
                batch[isentence][iword] = word2vec.get_vector(word) if word and word in word2vec else self.empty_embedding
        return tf.Variable(batch, dtype=tf.float32), []

    def _helper_sentence_stats(self):
        """"
        Get some information about the sentences.
        """
        words = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/SOStr.txt'), sep=' ', header=None)
        words.columns = ['wordstring']  # single column
        max_length = 0
        max_length_idx = 0
        avg_length = 0
        for sentence_index, row in words.iterrows():
            length = len(row.wordstring.split('|'))
            avg_length += length
            if length > max_length:
                max_length = length
                max_length_idx = sentence_index
        avg_length /= len(words)
        return max_length, max_length_idx, avg_length

    def _helper_standard_size(self, row):
        """
        Add additional placeholders so that a sentence parses to a fixed number of words (upper bound of all sentences)

        @requires
            wordstring must exist in the row

        @use
            Helper function for processing dataframe
        """
        items = row.wordstring.split('|')
        items.extend(['' for i in range(self.max_words - len(items))])
        return '|'.join(items)

    def train(self, batch_size=24, iterations=10000, lstm_count=64):
        """
        Build and run a tensorflow classifier on the sentiment data.

        @requires
            Google news word vectors are downloaded by user and specified at initialization (not synced due to size)
        """
        # Load and pre-process data
        print('Loading raw data...')
        labeled_sentences = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/labeled_sentences.txt'), sep='\t')
        # Limit to only training rows
        training_sentences = labeled_sentences[labeled_sentences['splitset_label'] == 1]
        # Parse sentences to individual words
        word_matrix, sentence_labels = self._build_word_matrix(training_sentences)

        # Load the pre-trained word vectors
        #   REF (download): https://code.google.com/archive/p/word2vec/
        #   REF (load): http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
        print('Loading word2vec model...')
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)

        print('Initializing Tensorflow model...')
        num_classes = len(self.class_labels)

        # Placeholders
        tf.reset_default_graph()
        labels = tf.placeholder(tf.float32, [batch_size, num_classes])
        batch = tf.placeholder(tf.float32, [batch_size, self.max_words, self.embedding_size])

        # LSTM layer
        lsmtm = tf.contrib.rnn.BasicLSTMCell(lstm_count)
        lstm_outputs, state = tf.nn.dynamic_rnn(lsmtm, batch, dtype=tf.float32)

        # Weights and bias for training
        weights = tf.Variable(tf.truncated_normal([lstm_count, num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        last_iter = tf.gather(lstm_outputs, int(lstm_outputs.get_shape()[0]) - 1)

        # Evaluation
        prediction = (tf.matmul(last_iter, weights) + bias)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # TODO - connect max pooling layer
		pool = tf.layers.max_pooling2d(inputs=batch, pool_size=[2,2], strides =2)

        # Initialize
        init = tf.global_variables_initializer()
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(init)

        # Run the training
        print('Running Tensorflow model...')
        for i in range(iterations):
            # Run a batch
            batch, labels = self._get_batch(word2vec, word_matrix, sentence_labels, batch_size)
            sess.run(optimizer, {'input_data': batch, 'labels': labels})

            # TODO - run pooling
			pool = tf.layers.max_pooling2d(inputs=batch, pool_size=[2,2], strides =2)

            # Save the network every 10,000 training iterations
            if i % 10000 == 0 and i:
                print('Saving iteration of model...')
                saver.save(sess, os.path.join(self.base_folder, "models/base_model.ckpt"), global_step=i)
        print('Done!')
