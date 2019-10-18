from typing import Set, Dict, Any

import multiprocessing
from nltk.corpus import brown, webtext, stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from progress.bar import Bar
import pyprind
import sys
import time


WORD2VEC_WINDOW_SIZE = 10
WORD2VEC_MIN_COUNT = 5
WORD2VEC_VECTOR_SIZE = 150
stopWords = set(stopwords.words('english'))


# This function gets as input vectors and returns the middle of each cluster that our clustering algorithm found.
# It is used to find the representing vectors for the splitting of each word.
# In order to use OPTICS you might need to do the following:
# 1. Open anaconda prompt
# 2. Type the following:
#       >> pip uninstall scikit-learn
#       >> pip uninstall sklearn
#       >> pip install sklearn
def cluster_word2vec_vectors(word_vector_array):
    from sklearn.cluster import OPTICS
    optics_model = OPTICS(min_samples=3)
    optics_model.fit(word_vector_array)   # training the model
    clusters_avg_vectors = []
    for cluster in optics_model.cluster_hierarchy_:
        avg_vector = np.zeros(WORD2VEC_VECTOR_SIZE)
        for vector_i in cluster:
            avg_vector = avg_vector + word_vector_array[vector_i]
        avg_vector = avg_vector / len(cluster)  # getting the avg of all the vectors in the cluster.
        clusters_avg_vectors.append(avg_vector)
    return clusters_avg_vectors


# This function returns the index of the closest vector to 'vector' from 'vector_group' in string format
def closest_vector(vector, vector_group):
    from scipy import spatial
    tree = spatial.KDTree(vector_group)
    return str(tree.query(vector))


# This function returns a trained Word2Vec classifier.
# and a the list of words in the corpus not including stopwords.
def word2vec_feat(corpus):
    sentences_list = []
    bar_max = len(corpus)
    bar = pyprind.ProgBar(bar_max, title='Creating Word2Vec Model', stream=sys.stdout, width=90)
    for sent in corpus:
        lowered_sent = []
        for w in sent:
            if w in stopWords:
                continue
            lowered_sent.append(w.lower())
        sentences_list.append(lowered_sent)
        bar.update()
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=WORD2VEC_MIN_COUNT, window=WORD2VEC_WINDOW_SIZE, sample=6e-5, alpha=0.03,
                         min_alpha=0.0007, negative=20, workers=cores - 1, size=WORD2VEC_VECTOR_SIZE)
    w2v_model.build_vocab(sentences_list)
    w2v_model.train(sentences_list, total_examples=w2v_model.corpus_count, epochs=30)
    # w2v_model.init_sims(replace=True)  # makes the vocabulary memory efficient.
    # words = set(w2v_model.wv.vocab.keys()) - set(stopwords.words('english'))
    return w2v_model


class WordSplitter:
    def __init__(self, corpus):
        sentences_list = []
        bar_max = 2 * len(corpus)
        bar = pyprind.ProgBar(bar_max, title='Processing corpus', stream=sys.stdout, width=90)
        for sent in corpus:
            lowered_sent = []
            for w in sent:
                if w in stopWords:
                    continue
                lowered_sent.append(w.lower())
            sentences_list.append(lowered_sent)
            bar.update()
        self.corpus = sentences_list
        word_counters = dict()
        for sentence_index, sentence in enumerate(self.corpus):
            for w in sentence:
                if w in stopWords:
                    continue
                if w not in word_counters:
                    word_counters[w] = 1
                else:
                    word_counters[w] = word_counters[w] + 1
            bar.update()
        self.word_counters = word_counters
        w2v_m = word2vec_feat(self.corpus)
        self.word2vec_model = w2v_m
        word2sentences, sentence2vector = self.__create_corpus_mapping()
        self.__word2sentences = word2sentences
        self.__sentence2vector = sentence2vector
        self.new_corpus = None
        self.new_w2v_model = None

    # This is a pre-processing function. It is used to create a mapping between every word in the corpus to the
    # sentences it appears in and also creates a word2vec vector for each sentence in the corpus.
    def __create_corpus_mapping(self):
        word2sentences = dict()
        sentence2vector = []
        word2vec_model = self.word2vec_model
        bar_max = len(self.corpus)
        bar = pyprind.ProgBar(bar_max, title='Creating Corpus Mapping', stream=sys.stdout, width=90)
        for w in self.corpus[0]:
            if w in stopWords:
                continue
            if self.word_counters[w] < WORD2VEC_MIN_COUNT:
                continue
            wor = w
        for sentence_index, sentence in enumerate(self.corpus):
            bar.update()
            # print("processing corpus mapping for sentence number " + str(sentence_index) + ":"
            # + str(len(self.corpus)))
            sentence_vector = np.zeros(WORD2VEC_VECTOR_SIZE)
            for w in sentence:
                if w in stopWords:
                    continue
                if self.word_counters[w] < WORD2VEC_MIN_COUNT:
                    continue
                sentence_vector = sentence_vector + np.array(word2vec_model[w])
                if w not in word2sentences:
                    word2sentences[w] = {sentence_index}
                else:
                    word2sentences[w].add(sentence_index)
            sentence2vector.append(sentence_vector)
        return word2sentences, sentence2vector

    # This function is used to find the center of each different representation on a word's cluster.
    # The function returns a dictionary containing all the center of the clusters generated for each word.
    def __create_cluster_vectors(self):
        bar_max = len(self.__word2sentences)
        bar = pyprind.ProgBar(bar_max, title='Clustering Word Vectors', stream=sys.stdout, width=90)
        word2split_vec_dict = dict()
        # counter = 0
        for word in self.__word2sentences:
            bar.update()
            # counter = counter + 1
            if word in stopWords:
                continue
            if self.word_counters[word] < WORD2VEC_MIN_COUNT:
                continue
            word_vector_array = []
            for sentence_i in self.__word2sentences[word]:
                word_vector_array.append(self.__sentence2vector[sentence_i])
            if len(word_vector_array) > 1000 or len(word_vector_array) < WORD2VEC_MIN_COUNT:
                continue
            # print(word)
            # print(str(len(word_vector_array)))
            # print(self.word_counters[word])
            word2split_vec_dict[word] = cluster_word2vec_vectors(word_vector_array)
            # print(str(counter) + "/" + str(len(self.__word2sentences)) + " | " + "appearances to clusters for " + word
            #       + " :\t\t\t" + str(len(word_vector_array)) + " : " + str(len(word2split_vec_dict[word])))
        return word2split_vec_dict

    # This function creates a new corpus in which the words have extensions according to the cluster they belong to.
    # def create_new_language_corpus(self):
    #     new_language_corpus = []
    #     w2v_model = self.word2vec_model
    #     word2split_vec_dict = self.__create_cluster_vectors()
    #     bar_max = len(self.corpus)
    #     bar = pyprind.ProgBar(bar_max, title='Creating New Language', stream=sys.stdout, width=90)
    #     for sentence_index, sentence in enumerate(self.corpus):
    #         # print("processing corpus mapping for sentence number " + str(sentence_index) + ":" +
    #         # str(len(self.corpus)))
    #         new_language_sentence = []
    #         for w in sentence:
    #             if w in stopWords:
    #                 continue
    #             if self.word_counters[w] < WORD2VEC_MIN_COUNT:
    #                 continue
    #             if w not in word2split_vec_dict:
    #                 continue
    #             word_w2v_vector = np.array(w2v_model[w])
    #             word_cluster_vectors = word2split_vec_dict[w]
    #             new_language_word = w + '_' + closest_vector(word_w2v_vector, word_cluster_vectors)
    #             new_language_sentence.append(new_language_word)
    #         new_language_corpus.append(new_language_sentence)
    #         bar.update()
    #     self.new_corpus = new_language_corpus
    #     return new_language_corpus

    # This function creates a new W2V model according to the new language we created.
    def create_new_language_w2v(self):
        new_language_word2vec_model = word2vec_feat(self.new_corpus)
        self.new_w2v_model = new_language_word2vec_model
        return new_language_word2vec_model

    # This function gets as input text in the old language and returns text in the new language.
    def classify(self, text):
        new_language_text = []
        w2v_model = self.word2vec_model
        word2split_vec_dict = self.__create_cluster_vectors()
        bar_max = len(self.corpus)
        bar = pyprind.ProgBar(bar_max, title='Changing old language to new language', stream=sys.stdout, width=90)
        for sentence_index, sentence in enumerate(text):
            new_language_sentence = []
            for w in sentence:
                if w in stopWords:
                    continue
                if self.word_counters[w] < WORD2VEC_MIN_COUNT:
                    continue
                if w not in word2split_vec_dict:
                    continue
                word_w2v_vector = np.array(w2v_model[w])
                word_cluster_vectors = word2split_vec_dict[w]
                new_language_word = w + '_' + closest_vector(word_w2v_vector, word_cluster_vectors)
                new_language_sentence.append(new_language_word)
            new_language_text.append(new_language_sentence)
            bar.update()
        return new_language_text


def plot_close_words(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show(block=False)
    input("Press [enter] to continue.")


if __name__ == '__main__':
    from nltk.corpus import brown
    print('Start')
    print(stopWords)
    brown_corpus = brown.sents()
    print(brown_corpus)
    print('Before creating word_splitter')
    word_splitter = WordSplitter(corpus=brown_corpus)
    print('After creating word_splitter')
    old_word2vec_model = word_splitter.word2vec_model
    print('Before creating new language corpus')
    new_corpus = word_splitter.create_new_language_corpus()
    print('After creating new language corpus')
    print('Before creating new word2vec')
    new_word2vec_model = word2vec_feat(new_corpus)
    print('After creating new  word2vec')
    print(new_corpus)
    print('Plotting:')
    test_word = "computer"
    plot_close_words(old_word2vec_model, test_word)
    plot_close_words(new_word2vec_model, test_word)
