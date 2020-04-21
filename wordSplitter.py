import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import nltk
import joblib
import signal
import multiprocessing
from multiprocessing import Pool
from gensim.corpora.wikicorpus import WikiCorpus
from functools import partial
import os.path
import enchant
import logging
import sys
import random
import string
import warnings
from sklearn.cluster import OPTICS
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
us = enchant.Dict("en_US")
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stopWords = set(stopwords.words('english'))

letter_array = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
NORMAL_DOC2VEC_NAME = 'WordSplitter.Doc2Vec.normal.pickle'
CLUSTERED_DOC2VEC_NAME = 'WordSplitter.Doc2Vec.clustered.pickle'
WIKIPEDIA_DUMP = "enwiki-latest-pages-articles.xml.bz2"
PICKLE_WIKICORPUS = "pickle_files/streaming_WikiCorpus.pickle"
PICKLE_CANONIC2DOC = "pickle_files/WordSplitter.CanonicToDoc.pickle"
PICKLE_CANONICBYLETTER = "pickle_files/WordSplitter.CanonicWordDict"
PICKLE_CANONICVECTORS = "pickle_files/WordSplitter.CanonicVectors"
PICKLE_CLUSTERBYLETTER = "pickle_files/WordSplitter.CanonicClusterDict"
PICKLE_EVENCLUSTER_FILENAME = "pickle_files/WordSplitter.EvenDict.WordToVectors."
PICKLE_WORDMAPPING_FILENAME = "pickle_files/WordSplitter.EvenDict.WordMapping.pkl"

VECTOR_SIZE = 256
WINDOW_SIZE = 8
MIN_COUNT = 19
OPTICS_MAX_DATA = 200000
OPTICS_CLUSTERING_FACTOR = 7
PCA_VAR = 0.75


def statistics(file_name):
    double_letter_set = []
    for l1 in letter_array:
        for l2 in letter_array:
            double_letter_set.append(l1 + l2)

    logger.info("STATISTICS: starting to create statistics ")
    counter_words = 0
    counter_10000 = 0
    counter_100000 = 0
    counter_500000 = 0
    counter_1000000 = 0
    counter_too_big = 0

    for letters in double_letter_set:
        file = file_name + '.' + letters + '.pickle'
        dictionary = pickle.load(open(file, "rb"))
        for key in dictionary.keys():
            length = len(dictionary[key])
            if length < 10000:
                continue
            elif length < 100000:
                counter_10000 += 1
            elif length < 500000:
                counter_100000 += 1
            elif length < 1000000:
                counter_500000 += 1
            elif length < 10000000:
                counter_1000000 += 1
            else:
                counter_too_big += 1
            counter_words += 1

    logger.info("STATISTICS: total words that can be clustered :               %i ", counter_words)
    logger.info("STATISTICS: words with vectors sizes of [10000, 100000) :     %i ", counter_10000)
    logger.info("STATISTICS: words with vectors sizes of [100000, 500000) :    %i ", counter_100000)
    logger.info("STATISTICS: words with vectors sizes of [500000, 1000000) :   %i ", counter_500000)
    logger.info("STATISTICS: words with vectors sizes of [1000000, 10000000) : %i ", counter_1000000)
    logger.info("STATISTICS: words with vectors sizes of [10000000, inf) :     %i ", counter_too_big)


def init_to_ignore_interrupt():
    """Should only be used when master is prepared to handle termination of child processes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def closest_vector(vector, vector_group):
    """
    This function calculates the closest vector to a given vector group.
    :param vector: the vector we are comparing to.
    :param vector_group: the group of vectors we want to compare to.
    :return: the index in vector_group of the closest vector to vector.
    """
    from scipy import spatial
    tree = spatial.KDTree(vector_group)
    dist, index = tree.query(vector)
    return str(index)


def pca_reduction(vectors, pca_var=0.85):
    from sklearn.decomposition import PCA

    pca_model = PCA(n_components=pca_var, svd_solver='full')
    pca_model.fit(vectors)
    reduced_vectors = pca_model.transform(vectors)
    return reduced_vectors


def get_clusters_vectors(canonic_vector_array, vector_size, optics_cluster_factor, pca_var):
    """
    This function creates the cluster vectors used to create the new corpus.
    Each cluster is represents a new word split and is represented by a vector.
    :param canonic_vector_array: a dictionary containing all the canonic words mapping to the normal words found in the
     corpus.
    :param vector_size: a parameter representing the size of the doc2vec vector.
    :param optics_cluster_factor: a parameter used to determine the minimum size of a cluster.
    :return: an array holding the vectors representing the new word splits.
    """
    min_samples = round(len(canonic_vector_array) / optics_cluster_factor)
    reduced_vectors = pca_reduction(canonic_vector_array, pca_var)
    optics_model = OPTICS(min_samples=min_samples, max_eps=10)
    optics_model.fit(reduced_vectors)  # training the model
    clusters_avg_vectors = []
    clusters_avg_vectors_num = []
    clusters_avg_vectors_final = []
    if max(optics_model.labels_) > 0:   # only if we found more than 1 split!
        for i in range(max(optics_model.labels_) + 2):
            clusters_avg_vectors.append(np.zeros(vector_size))
            clusters_avg_vectors_num.append(0)
        for i, cluster_i in enumerate(optics_model.labels_):
            clusters_avg_vectors[cluster_i+1] = clusters_avg_vectors[cluster_i+1] + canonic_vector_array[i]
            clusters_avg_vectors_num[cluster_i+1] = clusters_avg_vectors_num[cluster_i+1] + 1

        # calculate average of vector.
        for vec_i, vec in enumerate(clusters_avg_vectors):
            if clusters_avg_vectors_num[vec_i] > 0:
                clusters_avg_vectors_final.append(clusters_avg_vectors[vec_i] / clusters_avg_vectors_num[vec_i])
    return clusters_avg_vectors_final


class TaggedWikiDocument(object):
    """
    This class is used to stream wikipedia articles one by one from disk. Notice this will result in a long processing
    time.
    """

    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
        self.table = str.maketrans('', '', string.punctuation)
        logger.info("TaggedWikiDocument: INIT DONE for TaggedWikiDocument")

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            doc = []
            for c in content:   # type: str
                if c in stopWords:
                    continue
                elif hasattr(c, 'decode'):
                    doc.append(c.decode("utf-8").translate(self.table))
                else:
                    doc.append(c.translate(self.table))

            yield TaggedDocument(doc, [title])


class ClusteringTaggedWikiDocument(object):
    """
    This class is used to stream wikipedia articles one by one from disk.
    Notice that this class is classifying words to their appropriate cluster according to doc2vec_module and
    canonical2split dictionary.
    """
    def __init__(self, wiki, pickle_files="WordSplitter.CanonicClucterDict"):
        self.wiki = wiki
        self.wiki.metadata = True
        self.pickle_files = pickle_files
        self.doc2vec = Doc2Vec.load(self.NORMAL_DOC2VEC_NAME, mmap='r')
        self.lemmatizer = WordNetLemmatizer()  # https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
        self.table = str.maketrans('', '', string.punctuation)
        logger.info("ClusteringTaggedWikiDocument: INIT DONE for ClusteringTaggedWikiDocument")

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            normal_doc2vec = self.doc2vec
            sentence_vector = np.array(normal_doc2vec.docvecs[title])
            normal_doc2vec = None

            doc = []
            for c in content:   # type: str
                if c in stopWords:
                    continue
                elif hasattr(c, 'decode'):
                    w = c.decode("utf-8").translate(self.table)
                else:
                    w = c.translate(self.table)

                new_language_word = w
                conic_w = self.lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w.lower()))
                if conic_w[0] in letter_array and conic_w[1] in letter_array:
                    canonic_letter_file = self.pickle_files + '.' + conic_w[0:1] + '.pickle'
                    split_dict = pickle.load(open(canonic_letter_file, "rb"))
                    if conic_w in split_dict:
                        word_cluster_vectors = split_dict[conic_w]
                        split_dict = None
                        closest_index = closest_vector(sentence_vector, word_cluster_vectors)
                        if closest_index != 0:
                            new_language_word = w + '_' + str(closest_index)

                doc.append(new_language_word)

            yield TaggedDocument(doc, [title])
        logger.info("ClusteringTaggedWikiDocument: Iteration Over ClusteringTaggedWikiDocument is Done")


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "S": wordnet.ADJ_SAT,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


def create_canonic2doc_dictionary(dictionary_name="WordSplitter.canonic2doc.pickle",
                                  wikicorpus_picklefile='pickle_files/streaming_WikiCorpus.pickle', dump_per=250000):
    """
    Builds a dictionary mapping between canonic words to documents they appear in.
    :param dump_per: after how many articles to print data.
    :param dictionary_name: the name of the pickle files to be produced.
    :param wikicorpus_picklefile: name of pickles WikiCorpus instance to be used.
    NOTICE: this function takes ~5 Hours to run.
    """
    wikicorpus = pickle.load(open(wikicorpus_picklefile, "rb"))
    wikicorpus.metadata = True
    lemmatizer = WordNetLemmatizer()
    table = str.maketrans('', '', string.punctuation)

    pickle_i = 0
    word2docs = dict()

    article_num = 0
    word_num = 0
    logger.info("CANONIC2DOC: processed 0 articles word2sentence dictionary has 0 canonic words ")
    for content, (page_id, title) in wikicorpus.get_texts():
        article_num += 1
        canonic_word_set = set()
        for w in content:   # type: str
            w.translate(table)
            word_num += 1
            if w in stopWords or not us.check(w):
                continue

            canonic_word = lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w.lower()))
            if canonic_word not in canonic_word_set:
                canonic_word_set.add(canonic_word)

        for canonic_w in canonic_word_set:
            if canonic_w not in word2docs:
                word2docs[canonic_w] = {title}
            elif title not in word2docs[canonic_w]:
                word2docs[canonic_w].add(title)

        if article_num % dump_per == 0:
            pickle_filename = dictionary_name + '.' + str(pickle_i)
            pickle.dump(word2docs, open(pickle_filename, 'wb'))
            logger.info(
                "CANONIC2DOC: dumped %s, containing articles %i-%i, total %i canonic words, processed %i words ",
                pickle_filename, dump_per * pickle_i, article_num - 1, len(word2docs), word_num
            )
            pickle_i += 1
            word2docs = dict()

    pickle_filename = dictionary_name + '.' + str(pickle_i)
    pickle.dump(word2docs, open(pickle_filename, 'wb'))
    logger.info(
        "CANONIC2DOC: dumped dictionary %s, containing articles %i-%i, total %i canonic words, processed %i words ",
        pickle_filename, dump_per * pickle_i, article_num - 1, len(word2docs), word_num
    )
    pickle_i += 1
    length_filename = dictionary_name + ".length"
    pickle.dump(pickle_i, open(length_filename, 'wb'))
    return pickle_i


def words_set_by_letters_double_mp_worker(input_file="WordSplitter.canonic2doc.pickle",
                                          output_file="WordSplitter.CanonicWordSet", letters='PROBLEM'):
    input_dict = pickle.load(open(input_file, "rb"))
    temp_dict = dict()
    for canonic_word in input_dict.keys():
        if len(canonic_word) < 2:
            continue
        if canonic_word[0] != letters[0] or canonic_word[1] != letters[1]:   # Not the letters for this process.
            continue
        else:   # The letters this process is handling.
            if canonic_word not in temp_dict:
                temp_dict[canonic_word] = set(input_dict[canonic_word])
            else:
                temp_dict[canonic_word].update(input_dict[canonic_word])
    input_dict = None

    added_keys = 0
    added_values = 0
    output_dict_name = output_file + '.' + letters + '.pickle'
    if os.path.isfile(output_dict_name):
        output_dict = pickle.load(open(output_dict_name, "rb"))
        for key in temp_dict.keys():
            added_values += len(temp_dict[key])
            if key not in output_dict:
                added_keys += 1
                output_dict[key] = set(temp_dict[key])
            else:
                output_dict[key].update(temp_dict[key])
        pickle.dump(output_dict, open(output_dict_name, 'wb'))
    else:   # In case this is the first iteration and no dictionary exists.
        pickle.dump(temp_dict, open(output_dict_name, 'wb'))

    return added_keys, added_values


def create_canonic_words_set_by_letters_double_mp(input_file="WordSplitter.canonic2doc.pickle", canonic2doc_num=19,
                                                  output_file="WordSplitter.CanonicWordSet"):
    logger.info("CANONICBYLETTER: starting to create canonic word list ")
    number_of_pool_workers = 2

    double_letter_set = []
    for l1 in letter_array:
        for l2 in letter_array:
            double_letter_set.append(l1 + l2)

    # Create The output files.
    for i in range(canonic2doc_num):
        logger.info("CANONICBYLETTER: starting to process file %i/%i ", i, canonic2doc_num)
        input_dict = input_file + '.' + str(i)

        if i in [2, 5, 20]:
            number_of_pool_workers += 1
        processes = Pool(number_of_pool_workers)
        mpFunction = partial(words_set_by_letters_double_mp_worker, input_dict, output_file)
        results = processes.map(mpFunction, double_letter_set)
        processes.close()
        processes.join()

        new_words = sum([result[0] for result in results])
        new_titles = sum([result[1] for result in results])
        logger.info(
            "CANONICBYLETTER: finished processing file %i/%i, dictionaries gained: %i words, %i titles ",
            i, canonic2doc_num, new_words, new_titles
        )
    output_file_done = output_file + ".done"
    pickle.dump("DONE", open(output_file_done, 'wb'))

    logger.info("CANONICBYLETTER: finished processing all files ")


def remove_redundant_data_mp(filename="WordSplitter.CanonicWordSet", letters='PROBLEM'):
    file = filename + '.' + letters + '.pickle'
    dictionary = pickle.load(open(file, "rb"))
    keys_to_remove = set()
    for canonic_word in dictionary.keys():
        if len(dictionary[canonic_word]) < 10000:
            keys_to_remove.add(canonic_word)

    total_removed = len(keys_to_remove)
    for key in keys_to_remove:
        dictionary.pop(key, None)

    pickle.dump(dictionary, open(file, 'wb'))
    return total_removed


def remove_redundent_data(process_file):
    logger.info("REMOVE_REDUNDENT: starting to remove redundent words")

    double_letter_set = []
    for l1 in letter_array:
        for l2 in letter_array:
            double_letter_set.append(l1 + l2)
    redundant_removal_processes = Pool(10)

    mpFunction = partial(remove_redundant_data_mp, process_file)
    removed_words = redundant_removal_processes.map(mpFunction, double_letter_set)
    redundant_removal_processes.close()
    redundant_removal_processes.join()
    logger.info("REMOVE_REDUNDENT: total redundant words removed from letters dictionaries: %i ", sum(removed_words))


def split_evenly(process_files, even_clusters_filename="pickle_files/WordSplitter.EvenDict.WordToVectors.",
                 word_mapping_filename="pickle_files/WordSplitter.EvenDict.WordMapping.pkl", split_files=100):
    logger.info("SPLIT_EVENLY: starting to split data to even files ")

    double_letter_set = []
    for l1 in letter_array:
        for l2 in letter_array:
            double_letter_set.append(l1 + l2)

    total_pickle_size = 0
    for letters in double_letter_set:
        file = process_files + '.' + letters + '.pickle'
        dictionary = pickle.load(open(file, "rb"))
        total_pickle_size += sys.getsizeof(pickle.dumps(dictionary))

    even_dict_size = round(total_pickle_size / split_files)
    logger.info("SPLIT_EVENLY: even dictionary size: %i [Bytes]", even_dict_size)

    new_dict = dict()
    word_mapping = dict()
    dict_num = 0
    tot_words = 0
    tot_titles = 0
    for letters in double_letter_set:
        file = process_files + '.' + letters + '.pickle'
        dictionary = pickle.load(open(file, "rb"))
        for word in dictionary.keys():
            word_mapping[word] = dict_num
            new_dict[word] = dictionary[word]
            tot_words += 1
            tot_titles += len(dictionary[word])
            if sys.getsizeof(pickle.dumps(new_dict)) >= even_dict_size:
                logger.info(
                    "SPLIT_EVENLY: dumping dict %i, containing %i words, %i titles, %i bytes ",
                    dict_num, tot_words, tot_titles, sys.getsizeof(pickle.dumps(new_dict))
                )
                dict_filename = even_clusters_filename + str(dict_num)
                pickle.dump(new_dict, open(dict_filename, 'wb'))
                new_dict = dict()
                dict_num += 1
                tot_words = 0
                tot_titles = 0

    logger.info(
        "SPLIT_EVENLY: dumping dict %i, containing %i words, %i titles, %i bytes ",
        dict_num, tot_words, tot_titles, sys.getsizeof(pickle.dumps(new_dict))
    )
    dict_filename = even_clusters_filename + str(dict_num)
    pickle.dump(new_dict, open(dict_filename, 'wb'))
    dict_num += 1

    logger.info("SPLIT_EVENLY: dumping word mapping to even files str(word) -> int(file_number)")
    pickle.dump(word_mapping, open(word_mapping_filename, 'wb'))

    dict_length_filename = even_clusters_filename + "length"
    pickle.dump(dict_num, open(dict_length_filename, 'wb'))
    return dict_num


class EvenClusteringIterator(object):
    def __init__(self, pickle_file=None, max_data=250000):
        self.dictionary = pickle.load(open(pickle_file, "rb")) if pickle_file is not None else None
        self.doc2vec = Doc2Vec.load(self.NORMAL_DOC2VEC_NAME, mmap='r')
        self.max_data = max_data
        logger.info(
            "EvenClusteringIterator: INIT DONE for %s - %i words to cluster ",
            pickle_file, len(self.dictionary.keys())
        )

    def __iter__(self):
        for canonic_word in self.dictionary.keys():
            temp_dict = dict()

            if len(self.dictionary[canonic_word]) > (self.max_data / 10):
                select_num = round(self.max_data / 10)
            else:
                select_num = round(len(self.dictionary[canonic_word]) / 10)
            titles_vectors = random.sample(self.dictionary[canonic_word], k=select_num)
            temp_dict[canonic_word] = [self.doc2vec.docvecs[title] for title in titles_vectors]

            yield temp_dict


def create_even_clusters_mp(vector_size=256, clustering_factor=7, pca_var=0.85, given_args=None):
    if given_args is None:
        logger.info("CLUSTERING WORKER: NULL VALUES GIVEN")
        return "None", "None"
    elif type(given_args) is not dict:
        logger.info("CLUSTERING WORKER: NOT DICTIONARY GIVEN")
        return "None", "None"
    else:
        for canonic_word in given_args.keys():  # only one key in dictionary
            start_time = time.time()
            cluster_vectors = get_clusters_vectors(given_args[canonic_word], vector_size, clustering_factor, pca_var)
            if len(cluster_vectors) > 1:
                logger.info(
                    "CLUSTERING WORKER: clustering word %s - DONE (%i) - %.2f [min] ",
                    canonic_word, len(cluster_vectors), (time.time()-start_time)/60
                )
                return canonic_word, cluster_vectors
            else:
                logger.info(
                    "CLUSTERING WORKER: clustering word %s - DONE (0) - %.2f [min] ",
                    canonic_word, (time.time()-start_time)/60
                )
                return canonic_word, "None"


def create_clusters_from_even_files(vector_size=256, min_count=19, max_data=250000, clustering_factor=7, pca_var=0.85,
                                    even_vectors_filename="WordSplitter.CanonicWordSet", number_of_even_files=0,
                                    cluster_by_letter_files="WordSplitter.CanonicClusterDict"):
    """
    Creates the word split clusters used for classifying wikipedia.
    After running this function the following variables are initiated:
        1. self.canonic2split_vectors
    """
    from multiprocessing import Pool
    logger.info("CLUSTERING: starting to create clusters for %i files ", number_of_even_files)
    cluster_start = time.time()

    for i in range(number_of_even_files):
        processes = Pool(2)
        file = even_vectors_filename + str(i)
        even_clustering_iterator = EvenClusteringIterator(file, max_data)
        worker_function = partial(create_even_clusters_mp, vector_size, clustering_factor, pca_var)
        results = processes.map(worker_function, even_clustering_iterator, chunksize=1)
        processes.close()
        processes.join()

        even_clusters_dict = dict()
        dump_file = cluster_by_letter_files + '.' + str(i)
        logger.info("CLUSTERING: preparing to dump file %i ", i)
        for canonic_word, cluster_vectors in results:
            if canonic_word is not "None" and cluster_vectors is not "None":
                even_clusters_dict[canonic_word] = cluster_vectors

        pickle.dump(even_clusters_dict, open(dump_file, 'wb'))

    cluster_end = time.time()
    logger.info(
        "CLUSTERING: finished creating clusters, total time: %s [hours]",
        str(round(((cluster_end - cluster_start) / 60) / 60))
    )


class WordSplitter:
    def __init__(self, vector_size, window_size, min_count, optics_max_data_size, optics_cluster_factor, pca_var,
                 wikicorpus_file, canonic2doc_file, canonicbyletter_file,
                 canonicvectors_file,
                 clusterbyletter_file,
                 even_clusters_file, word_mapping_file, normal_doc2vec_name, clustered_doc2vec_name):

        self.WikiCorpus = None
        self.corpus_length = None
        self.VECTOR_SIZE = vector_size
        self.WINDOW_SIZE = window_size
        self.MIN_COUNT = min_count
        self.OPTICS_MAX_DATA = optics_max_data_size
        self.OPTICS_CLUSTERING_FACTOR = optics_cluster_factor
        self.PCA_VARIANCE = pca_var
        self.PICKLE_WIKICORPUS = wikicorpus_file
        self.PICKLE_CANONIC2DOC = canonic2doc_file
        self.PICKLE_CANONIC2DOC_NUM = 0
        self.PICKLE_CANONICBYLETTER = canonicbyletter_file
        self.PICKLE_CANONICVECTORS = canonicvectors_file
        self.PICKLE_CLUSTERBYLETTER = clusterbyletter_file
        self.PICKLE_EVENCLUSTER_FILENAME = even_clusters_file
        self.PICKLE_WORDMAPPING_FILENAME = word_mapping_file
        self.NORMAL_DOC2VEC_NAME = normal_doc2vec_name
        self.CLUSTERED_DOC2VEC_NAME = clustered_doc2vec_name
        logger.info("INIT: Creating Word Splitter Module")
        logger.info("INIT: Creating Word Splitter Module")
        logger.info("INIT: Vector Size: %i ", self.VECTOR_SIZE)
        logger.info("INIT: Window Size: %i ", self.WINDOW_SIZE)
        logger.info("INIT: Minimum Count: %i ", self.MIN_COUNT)
        logger.info("INIT: Name of Pickle Dump for WikiCorpus Inst: %s ", self.PICKLE_WIKICORPUS)
        logger.info("INIT: Name of Pickle Dump for Canonic to Doc Dictionary: %s ", self.PICKLE_CANONIC2DOC)
        logger.info("INIT: Name of Pickle Dump for Canonic by Letter Set: %s ", self.PICKLE_CANONICBYLETTER)
        logger.info("INIT: Name of Pickle Dump for Canonic Vectors by Letters: %s ", self.PICKLE_CANONICVECTORS)
        logger.info("INIT: Name of Pickle Dump for Clusters by Letter Dictionary: %s ", self.PICKLE_CLUSTERBYLETTER)
        logger.info("INIT: Name of Normal DOC2VEC: %s ", self.NORMAL_DOC2VEC_NAME)
        logger.info("INIT: Name of Clustered DOC2VEC: %s ", self.CLUSTERED_DOC2VEC_NAME)
        logger.info("INIT: OPTICS Vector Size Limit: %i ", self.OPTICS_MAX_DATA)
        logger.info("INIT: OPTICS Clustering Factor: %i ", self.OPTICS_CLUSTERING_FACTOR)
        logger.info("INIT: PCA Minimum Variance: %.3f ", self.PCA_VARIANCE)
        logger.info("INIT: Highest Pickle Protocol: %i ", int(pickle.HIGHEST_PROTOCOL))

        self.taggedWikiDocument = None
        self.normal_doc2vec = None
        self.canonic2split_vectors = None

        self.clusteringTaggedWikiDocument = None
        self.clustered_doc2vec = None
        self.even_files_num = 0

    def preprocess(self, create_canonic2doc=False, create_canonic_by_letter=False, create_even_files=False,
                   create_clusters_by_letter=False):
        """
        Builds doc2vec initial vocabulary.
        After running this function the following variables are initiated:
            1. self.normal_doc2vec
            2. self.taggedWikiDocument
        """
        logger.info("WORD SPLITTER: Starting pre-processing")
        preprocess_start = time.time()

        if os.path.isfile(self.PICKLE_WIKICORPUS):
            logger.info("PREPROCESS: existing WikiCorpus Found ")
            self.WikiCorpus = pickle.load(open(self.PICKLE_WIKICORPUS, "rb"))
        else:
            self.WikiCorpus = WikiCorpus(WIKIPEDIA_DUMP, article_min_tokens=self.MIN_COUNT*5,
                                         lower=True)
            self.WikiCorpus.metadata = True
            self.corpus_length = self.WikiCorpus.length
            pickle.dump(self.WikiCorpus, open(self.PICKLE_WIKICORPUS, 'wb'))

        if create_canonic2doc:
            self.PICKLE_CANONIC2DOC_NUM = create_canonic2doc_dictionary(dictionary_name=self.PICKLE_CANONIC2DOC, wikicorpus_picklefile=self.PICKLE_WIKICORPUS, dump_per=150000)
        else:
            canonic2doc_num_filename = self.PICKLE_CANONIC2DOC + ".length"
            if os.path.isfile(canonic2doc_num_filename):
                logger.info("PREPROCESS: existing CanonicToDoc Files Found ")
                self.PICKLE_CANONIC2DOC_NUM = pickle.load(open(canonic2doc_num_filename, "rb"))
            else:
                logger.info("PREPROCESS: CanonicToDoc dictionary doesn't exist, please create in preprocess ")

        if create_canonic_by_letter:
            create_canonic_words_set_by_letters_double_mp(input_file=self.PICKLE_CANONIC2DOC,
                                                          canonic2doc_num=self.PICKLE_CANONIC2DOC_NUM,
                                                          output_file=self.PICKLE_CANONICBYLETTER)
            statistics(self.PICKLE_CANONICBYLETTER)
        else:
            canonic_by_letters_filename = self.PICKLE_CANONICBYLETTER + ".done"
            if os.path.isfile(canonic_by_letters_filename):
                logger.info("PREPROCESS: existing CanonicWordDict by letters Files Found ")
            else:
                logger.info("PREPROCESS: CanonicWordDict dictionaries don't exist, please create in preprocess ")

        # remove_redundent_data(self.PICKLE_CANONICBYLETTER)
        if create_even_files:
            self.even_files_num = split_evenly(self.PICKLE_CANONICBYLETTER,
                                               even_clusters_filename=self.PICKLE_EVENCLUSTER_FILENAME,
                                               word_mapping_filename=self.PICKLE_WORDMAPPING_FILENAME, split_files=500)
        else:
            even_files_num_filename = self.PICKLE_EVENCLUSTER_FILENAME + "length"
            if os.path.isfile(even_files_num_filename):
                logger.info("PREPROCESS: existing even canonic words dictionaries found ")
                self.even_files_num = pickle.load(open(even_files_num_filename, "rb"))
            else:
                logger.info("PREPROCESS: Even canonic words dictionaries don't exist, please create in preprocess ")

        self.taggedWikiDocument = TaggedWikiDocument(self.WikiCorpus)
        if os.path.isfile(self.NORMAL_DOC2VEC_NAME):
            logger.info("PREPROCESS: existing Normal Doc2vec Model Found ")
            self.normal_doc2vec = Doc2Vec.load(self.NORMAL_DOC2VEC_NAME, mmap='r')
        else:
            self.normal_doc2vec = Doc2Vec(dm=0, dbow_words=1, vector_size=self.VECTOR_SIZE, window=self.WINDOW_SIZE,
                                          min_count=self.MIN_COUNT, epochs=1, workers=multiprocessing.cpu_count()-1,
                                          docvecs_mapfile="normal_mapfile")
            self.normal_doc2vec.build_vocab(self.taggedWikiDocument, progress_per=500000)
            self.normal_doc2vec.train(documents=self.taggedWikiDocument, total_examples=self.normal_doc2vec.corpus_count,
                                      epochs=self.normal_doc2vec.epochs, report_delay=1800)
            self.normal_doc2vec.save(self.NORMAL_DOC2VEC_NAME)

        if create_clusters_by_letter:
            create_clusters_from_even_files(vector_size=self.VECTOR_SIZE, min_count=self.MIN_COUNT,
                                            max_data=self.OPTICS_MAX_DATA,
                                            clustering_factor=self.OPTICS_CLUSTERING_FACTOR,
                                            pca_var=self.PCA_VARIANCE,
                                            even_vectors_filename=self.PICKLE_EVENCLUSTER_FILENAME,
                                            number_of_even_files=self.even_files_num,
                                            cluster_by_letter_files=self.PICKLE_CLUSTERBYLETTER)

        preprocess_end = time.time()
        logger.info("PREPROCESS: total preprocess time: %s [hours]",
                    str(round(((preprocess_end - preprocess_start) / 60) / 60)))

    def train(self):
        """
        Classifies wikipedia.
        After running this function the following variables are initiated:
            1. self.clusteringTaggedWikiDocument
            2. self.clustered_doc2vec
        self.clustered_doc2vec is the final doc2vec trained on the whole split wikipedia
        """
        train_start = time.time()

        self.clusteringTaggedWikiDocument = ClusteringTaggedWikiDocument(self.WikiCorpus,
                                                                         pickle_files=self.PICKLE_CLUSTERBYLETTER)
        self.clustered_doc2vec = Doc2Vec(dm=0, dbow_words=1, vector_size=self.VECTOR_SIZE, window=self.WINDOW_SIZE,
                                         min_count=self.MIN_COUNT, epochs=2, workers=multiprocessing.cpu_count()-1,
                                         docvecs_mapfile="split_mapfile")
        self.clustered_doc2vec.build_vocab(self.clusteringTaggedWikiDocument, progress_per=50000)
        self.clustered_doc2vec.train(documents=self.clusteringTaggedWikiDocument,
                                     total_examples=self.clustered_doc2vec.corpus_count,
                                     epochs=self.clustered_doc2vec.epochs, report_delay=1800)
        self.clustered_doc2vec.save(self.CLUSTERED_DOC2VEC_NAME)
        train_end = time.time()
        logger.info("TRAIN: total training time: %s [hours]", str(round(((train_end - train_start) / 60) / 60)))


if __name__ == '__main__':
    word_splitter = WordSplitter(VECTOR_SIZE, WINDOW_SIZE, MIN_COUNT, OPTICS_MAX_DATA, OPTICS_CLUSTERING_FACTOR,
                                 PCA_VAR, PICKLE_WIKICORPUS, PICKLE_CANONIC2DOC, PICKLE_CANONICBYLETTER, PICKLE_CANONICVECTORS,
                                 PICKLE_CLUSTERBYLETTER, PICKLE_EVENCLUSTER_FILENAME, PICKLE_WORDMAPPING_FILENAME,
                                 NORMAL_DOC2VEC_NAME, CLUSTERED_DOC2VEC_NAME)

    word_splitter.preprocess(create_canonic2doc=False, create_canonic_by_letter=False, create_even_files=False,
                             create_clusters_by_letter=True)
    joblib.dump(word_splitter, open('word_splitter_preprocess.pickle', 'wb'), protocol=4, compress=9)

    word_splitter.train()
    joblib.dump(word_splitter, open('word_splitter_trained.pickle', 'wb'), protocol=4, compress=9)
