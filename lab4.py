import gensim
import numpy as np
import os
import pandas as pd


# from lab4 import *; sdl = SentimentDeepLearner(); df = sdl._prepare_data(sentence_from_scratch=False, vectors_from_scratch=True)
class SentimentDeepLearner(object):

    def __init__(self, word2vec_path='D:/Temp/GoogleNews-vectors-negative300.bin'):
        self.word2vec_path = word2vec_path
        self.base_folder = os.path.realpath(os.getcwd())
        self.class_labels = ['strongly negative', 'negative', 'neutral', 'positive', 'strongly positive']
        self.max_words = 56  # maximum number of words in a sentence (from analysis of SOStr.txt)
        self.embedding_size = 300  # from Google word2vec

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
        words['wordstring'] = words.apply(self._standard_size, axis=1)  # regularize sentence length
        data = data.join(words)

        # 8. Drop "dev" labeled rows (this saves time on embedding)
        #   reduce rows from 11,855 to 10,243
        data = data.drop(data[data['splitset_label'] == 3].index)

        # 9. Save results for future use
        #   [splitset_label, sentiment, sentiment_label, wordstring]
        data = data.drop(columns=['sentence'])
        data.to_csv(os.path.join(self.base_folder, 'sentimentData/labeled_sentences.txt'), header=True, index=False, sep='\t')
        return data

    def _build_sentence_vectors(self, data):
        """
        Create a datset which contains the word vectors of all the sentences.

        @use
            This should be called after self._build_sentence_dataset
            Loads the word2vec model and extracts vectors to save, so the model doesn't have to stay in memory
            Sentences (10,243), MaxSentence (56 words), EmbeddingArray (300) = 172M cells, 10,243 x 16,800

        """
        sentence_count = len(data)
        empty = [0 for i in range(self.embedding_size)]  # no word, or word not in word2vec

        # Load the word2vec model (this takes a couple minutes)
        #   REF (download): https://code.google.com/archive/p/word2vec/
        #   REF (load): http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
        print('Loading word2vec model...')
        model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)

        # Build out 16,800 columns for each sentence from word vectors
        print('Building sentence vectors...')
        sentence_vectors = np.ndarray(shape=(sentence_count, self.max_words * self.embedding_size))
        for sentence_index, row in data.iterrows():
            words = row.wordstring.split('|')
            # Concatenate all word vectors for sentence (using empty if not in model or empty word)
            sentence_vector = np.concatenate(
                [(model.get_vector(word) if word and word in model else empty) for word in words],
                axis=0
            )
            sentence_vectors[sentence_index] = np.transpose(sentence_vector)  # convert column vector to row

        # Create dataframe
        print('Building dataframe...')
        columns = []
        for i in range(self.max_words):  # each word
            for j in range(self.embedding_size):  # each embedding dimension for the word
                columns.append('{0}.{1}'.format(i, j))
        df = pd.DataFrame(sentence_vectors, columns=columns)
        # NOTE: Easier to simply re-generate the frame as the file is over 1GB and takes a while to write
        # df.to_csv(os.path.join(self.base_folder, 'sentimentData/sentence_vectors.txt'), header=True, sep=',')
        return df

    def _prepare_data(self, sentence_from_scratch=False, vectors_from_scratch=True):
        """
        Pre-process data files
        """
        # 1. Get labeled sentences with word parse
        if sentence_from_scratch:
            labeled_sentences = self._build_labeled_sentences()
        else:
            labeled_sentences = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/labeled_sentences.txt'), sep='\t')

        # 2. Get word vectors for each sentence
        if vectors_from_scratch:
            # NOTE: if you want to save a file version, need to uncomment at the end of the function (warning LARGE)
            sentence_vectors = self._build_sentence_vectors(labeled_sentences)
        else:
            sentence_vectors = pd.read_csv(os.path.join(self.base_folder, 'sentimentData/sentence_vectors.txt'), sep=',')

        print(sentence_vectors.shape)

        # Split into train and test
        # train_data = data.loc[data['splitset_label'] == 1]
        # test_data = data.loc[data['splitset_label'] == 2]

    def _standard_size(self, row):
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

    def main(self):
        """

        @requires
            Google news word vectors are downloaded by user and specified at initialization (not synced due to size)
        """
        # Load pre-trained word vectors from Google word2vec (this can take a while)
        pass
