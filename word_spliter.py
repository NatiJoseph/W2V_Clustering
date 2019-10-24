from typing import Set, Dict, Any
import multiprocessing
from nltk.corpus import brown, webtext, stopwords
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from progress.bar import Bar
import pyprind
import sys
import time


WORD2VEC_WINDOW_SIZE = 5
WORD2VEC_MIN_COUNT = 20
WORD2VEC_VECTOR_SIZE = 350
DOC2VEC_MIN_COUNT = 15
DOC2VEC_WINDOW_SIZE = WORD2VEC_WINDOW_SIZE
DOC2VEC_VECTOR_SIZE = WORD2VEC_VECTOR_SIZE
OPTICS_MIN_CLUSTERING_FACTOR = 5
stopWords = set(stopwords.words('english'))


def print_dictionary_to_excel(word_splitter_inst, using_corpus):
    # from interface import using_corpus
    import xlwt
    from xlwt import Workbook

    words = word_splitter_inst.word2vec_model.wv.vocab.keys()
    words_2 = word_splitter_inst.word_counters
    words_3 = word_splitter_inst.new_w2v_model.wv.vocab.keys()
    bar_max = len(words_2.items()) + len(words) + len(words_3)
    bar = pyprind.ProgBar(bar_max, title='Creating W2V Vocab and Word Counters Excel Tables',
                          stream=sys.stdout, width=90)

    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    for i, word in enumerate(words):
        bar.update()
        sheet1.write(i, 0, str(word))
    name = "old_w2v_vocab" + using_corpus + ".xls"
    wb.save(name)

    wb_2 = Workbook()
    sheet2 = wb_2.add_sheet('Sheet 1')
    i = 0
    for word, word_counter in words_2.items():
        bar.update()
        sheet2.write(i, 0, str(word))
        sheet2.write(i, 1, str(word_counter))
        i = i + 1
    name2 = "word_counters" + using_corpus + ".xls"
    wb_2.save(name2)

    wb_3 = Workbook()
    sheet3 = wb_3.add_sheet('Sheet 1')
    i_2 = 0
    for i_2, word in enumerate(words_3):
        bar.update()
        sheet3.write(i_2, 0, str(word))
    name3 = "new_w2v_vocab" + using_corpus + ".xls"
    wb_3.save(name3)


def print_canonic_words_to_excel(word_splitter_inst, using_corpus):
    # from interface import using_corpus
    import xlwt
    from xlwt import Workbook

    conic_words_dict = word_splitter_inst.canonic_to_word
    bar_max = len(conic_words_dict.items())
    bar = pyprind.ProgBar(bar_max, title='Creating Conic Words Excel Tables', stream=sys.stdout, width=90)

    wb = Workbook()
    sheet = wb.add_sheet('Sheet 1')
    i = 0
    for conic_word, words in conic_words_dict.items():
        bar.update()
        sheet.write(i, 0, str(conic_word))
        for j, word in enumerate(words):
            j = j + 1
            sheet.write(i, j, str(word))
        i = i + 1
    name = "conic_words_dict" + using_corpus + ".xls"
    wb.save(name)


# This function gets as input vectors and returns the middle of each cluster that our clustering algorithm found.
# It is used to find the representing vectors for the splitting of each word.
# In order to use OPTICS you might need to do the following:
# 1. Open anaconda prompt
# 2. Type the following:
#       >> pip uninstall scikit-learn
#       >> pip uninstall sklearn
#       >> pip install sklearn
def cluster_word2vec_vectors(word_vector_array):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from sklearn.cluster import OPTICS
    min_samples = int(len(word_vector_array) / OPTICS_MIN_CLUSTERING_FACTOR)
    optics_model = OPTICS(min_samples=min_samples)
    optics_model.fit(word_vector_array)   # training the model
    clusters_avg_vectors = []
    clusters_avg_vectors_num = []
    # for cluster in optics_model.cluster_hierarchy_:
    #     print(cluster)
    #     avg_vector = np.zeros(WORD2VEC_VECTOR_SIZE)
    #     for vector_i in cluster:
    #         avg_vector = avg_vector + word_vector_array[vector_i]
    #     avg_vector = avg_vector / len(cluster)  # getting the avg of all the vectors in the cluster.
    #     clusters_avg_vectors.append(avg_vector)
    for i in range(max(optics_model.labels_) + 1):
        clusters_avg_vectors.append(np.zeros(WORD2VEC_VECTOR_SIZE))
        clusters_avg_vectors_num.append(0)
    for i, cluster_i in enumerate(optics_model.labels_):
        if cluster_i == -1:
            continue
        clusters_avg_vectors[cluster_i] = clusters_avg_vectors[cluster_i] + word_vector_array[i]
        clusters_avg_vectors_num[cluster_i] = clusters_avg_vectors_num[cluster_i] + 1
    clusters_avg_vectors_final = []
    for vec_i, vec in enumerate(clusters_avg_vectors):
        if clusters_avg_vectors_num[vec_i] > 0:
            clusters_avg_vectors_final.append(clusters_avg_vectors[vec_i] / clusters_avg_vectors_num[vec_i])
    return clusters_avg_vectors_final


# This function returns the index of the closest vector to 'vector' from 'vector_group' in string format
def closest_vector(vector, vector_group):
    from scipy import spatial
    tree = spatial.KDTree(vector_group)
    dist, index = tree.query(vector)
    return str(index)


# This function returns a trained Word2Vec classifier.
# and a the list of words in the corpus not including stopwords.
# We found out that because of the clustering process we are losing a lot of words to the min_count parameter, we
# decided that we have to lower the min_count after the clustering process.
def word2vec_feat(corpus, post_clustering=False):
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
    w2v_min_count = int(WORD2VEC_MIN_COUNT / 4) + 1 if post_clustering else WORD2VEC_MIN_COUNT
    w2v_model = Word2Vec(min_count=w2v_min_count, window=WORD2VEC_WINDOW_SIZE, sample=6e-5, alpha=0.03,
                         min_alpha=0.0007, negative=20, workers=cores-1, size=WORD2VEC_VECTOR_SIZE)
    w2v_model.build_vocab(sentences_list)
    w2v_model.train(sentences_list, total_examples=w2v_model.corpus_count, epochs=30)
    # w2v_model.init_sims(replace=True)  # makes the vocabulary memory efficient.
    # words = set(w2v_model.wv.vocab.keys()) - set(stopwords.words('english'))
    return w2v_model


# This function returns a trained Doc2Vec classifier.
def doc2vec_feat(corpus, post_clustering=False):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    cores = multiprocessing.cpu_count()
    d2v_model = Doc2Vec(documents, vector_size=DOC2VEC_VECTOR_SIZE, window=DOC2VEC_WINDOW_SIZE,
                        min_count=DOC2VEC_MIN_COUNT, workers=cores-1)
    return d2v_model


class WordSplitter:
    def __init__(self, corpus):
        lemmatizer = WordNetLemmatizer()    # https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
        sentences_list = []
        canonic_to_word = dict()
        print("INIT WORD SPLITTER STARTED")
        for sent in corpus:
            lowered_sent = []
            for w in sent:
                if w in stopWords:
                    continue
                # if w is "," or w is "." or w is"\\" or w is "#" or w is "1" or w is "0" or w is "2" or w is "3" \
                #         or w is "?" or w is "!" or w is ":" or w is "\'" or w is "-" or w is "!" or w is "\""\
                #         or w is "/" or w is "[" or w is "]"  or w is "(" or w is ")" or w is ".."  or w is "..."\
                #         or w is ">" or w is "<" or w is "=" or w is "*":
                if len(w) <= 1 or w is ".." or w is "..." or w is "()" or w is "->" or w is "<-" or w is "://" \
                        or w is "\"?" or w is "=\"" or w is "...\"" or w is "--" or w is "::_" or w is "*." \
                        or w is ".\"" or w is "\"." or w is "...." or w is ")." or w is "::" or w is "[@" \
                        or w is "?\"" or w is "(\"" or w is ")\"" or w is "!\"" or w is "\")" or w is "\"(" \
                        or w is ".)" or w is "\"," or w is "\\" or w.isdigit():
                    continue
                canonic_w = lemmatizer.lemmatize(w.lower())
                if canonic_w not in canonic_to_word:
                    canonic_to_word[canonic_w] = {w.lower()}
                else:
                    canonic_to_word[canonic_w].add(w.lower())
                lowered_sent.append(w.lower())
            sentences_list.append(lowered_sent)
        self.corpus = sentences_list
        self.canonic_to_word = canonic_to_word
        self.word_counters = self.__create_corpus_word_counters()
        w2v_m = word2vec_feat(self.corpus)
        self.word2vec_model = w2v_m
        # doc2vec processing
        d2v_m = doc2vec_feat(self.corpus)
        self.d2v_model = d2v_m
        word2sentences, sentence2vector = self.__create_corpus_mapping()
        self.__word2sentences = word2sentences
        self.__sentence2vector = sentence2vector
        w2p_vec_dict = self.__create_cluster_vectors()
        self.word2split_vec_dict = w2p_vec_dict
        n_corpus = self.create_new_language_corpus()
        self.new_corpus = n_corpus
        n_word2vec_model = self.create_new_language_w2v()
        self.new_w2v_model = n_word2vec_model
        n_d2v_model = doc2vec_feat(self.new_corpus)
        self.new_d2v_model = n_d2v_model
        print("INIT WORD SPLITTER ENDED")

    # This function creates a dictionary that contains the number of times each word appears in the corpus.
    def __create_corpus_word_counters(self):
        word_counters = dict()
        for sentence_index, sentence in enumerate(self.corpus):
            for w in sentence:
                if w in stopWords:
                    continue
                if w not in word_counters:
                    word_counters[w] = 1
                else:
                    word_counters[w] = word_counters[w] + 1
        return word_counters

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
            for w in sentence:
                if w in stopWords:
                    continue
                # if self.word_counters[w] < WORD2VEC_MIN_COUNT:
                #     continue
                if w not in word2sentences:
                    word2sentences[w] = {sentence_index}
                else:
                    word2sentences[w].add(sentence_index)
            sentence2vector.append(self.d2v_model.docvecs[sentence_index])
        return word2sentences, sentence2vector

    # This function is used to find the center of each different representation on a word's cluster.
    # The function returns a dictionary containing all the center of the clusters generated for each word.
    def __create_cluster_vectors(self):
        bar_max = len(self.__word2sentences)
        bar = pyprind.ProgBar(bar_max, title='Clustering Word Vectors', stream=sys.stdout, width=90)
        word2split_vec_dict = dict()
        for word in self.__word2sentences:
            bar.update()
            if word in stopWords:
                continue
            if self.word_counters[word] < WORD2VEC_MIN_COUNT:
                continue
            word_vector_array = []
            lemmatizer = WordNetLemmatizer()  # https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
            for same_canonic_word in self.canonic_to_word[lemmatizer.lemmatize(word)]:
                w = same_canonic_word if same_canonic_word in self.__word2sentences else word
                # print(word + " : " + same_canonic_word + " : " + w)
                for sentence_i in self.__word2sentences[w]:
                    word_vector_array.append(self.__sentence2vector[sentence_i])
            if len(word_vector_array) > 1500 or len(word_vector_array) < WORD2VEC_MIN_COUNT:
                # print("ERROR: word_vector_array length for the word \'" + word + "\' is: " +
                #       str(len(word_vector_array)))
                continue
            # print(word)
            # print(str(len(word_vector_array)))
            # print(self.word_counters[word])
            word2split_vec_dict[word] = cluster_word2vec_vectors(word_vector_array)
            # print("Total clusters for " + word + " : " + str(len(word2split_vec_dict[word])))
        return word2split_vec_dict

    # This function creates a new W2V model according to the new language we created.
    def create_new_language_w2v(self):
        new_language_word2vec_model = word2vec_feat(self.new_corpus, post_clustering=True)
        return new_language_word2vec_model

    # This function creates a new corpus according to the new language we created.
    def create_new_language_corpus(self):
        n_corpus = self.classify(self.corpus)
        return n_corpus

    # This function gets as input a word in the old language and returns the number of words created from that word
    # in the new language and the most similar word to it in the new language.
    def get_new_words(self, old_word):
        word2split_vec_dict = self.word2split_vec_dict
        new_w2v_model = self.new_w2v_model
        if old_word not in word2split_vec_dict:
            if old_word not in self.word_counters or old_word not in new_w2v_model.wv.vocab:
                print("ERROR: word " + str(old_word) + " doesn't exist in corpus")
                return old_word, None
            else:
                most_similar_word = new_w2v_model.wv.most_similar(positive=old_word, topn=1)
                most_similar_word = most_similar_word[0]
                most_similar_word = most_similar_word[0]
                return old_word, most_similar_word
        new_words = []
        print("The word " + old_word + " appears " + str(self.word_counters[old_word]) + " times in old corpus")
        most_similar_word_old = self.word2vec_model.wv.most_similar(positive=old_word, topn=1)
        most_similar_word_old = most_similar_word_old[0]
        most_similar_word_old = most_similar_word_old[0]
        print("the most similar word in the old language is: " + str(most_similar_word_old))
        print("The word " + old_word + " has " + str(len(self.word2split_vec_dict[old_word])) + " meanings")
        for word_index, word_vector in enumerate(word2split_vec_dict[old_word]):
            new_word = old_word + '_' + str(word_index)
            if new_word not in new_w2v_model.wv.vocab:
                continue
            most_similar_word = new_w2v_model.wv.most_similar(positive=new_word, topn=1)
            most_similar_word = most_similar_word[0]
            most_similar_word = most_similar_word[0]
            tup = new_word, most_similar_word
            new_words.append(tup)
        if len(new_words) == 0:
            print("no new words")
            most_similar_word = new_w2v_model.wv.most_similar(positive=old_word, topn=1)
            most_similar_word = most_similar_word[0]
            most_similar_word = most_similar_word[0]
            return old_word, most_similar_word
        return new_words

    # This function gets as input text in the old language and returns text in the new language.
    def classify(self, text):
        new_language_text = []
        d2v_model = self.d2v_model
        word2split_vec_dict = self.word2split_vec_dict
        bar_max = len(text)
        bar = pyprind.ProgBar(bar_max, title='Changing old language to new language', stream=sys.stdout, width=90)
        for sentence_index, sentence in enumerate(text):
            new_language_sentence = []
            sentence_d2v_vector = np.array(d2v_model.docvecs[sentence_index])
            for w in sentence:
                if w in stopWords:
                    continue
                if self.word_counters[w] < WORD2VEC_MIN_COUNT:
                    continue
                if w not in word2split_vec_dict:
                    new_language_word = w
                else:
                    word_cluster_vectors = word2split_vec_dict[w]
                    new_language_word = w + '_' + str(closest_vector(sentence_d2v_vector, word_cluster_vectors))
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
