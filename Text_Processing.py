from nltk.corpus import stopwords, brown
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize, pos_tag_sents
import re
import sys, time
import pickle
import codecs
from gensim.corpora import WikiCorpus
import pyprind
# sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace')
from MyWikiCorpus import MyWikiCorpus


def make_corpus(wiki_dump, file_name='wiki_corpus.txt'):
    """
                        This function is used to create a txt file containing the whole wikipedia dump.
                        Please notice this has a really long running time, and will also require a huge memory.
                        expected txt file size: 40-50 GB.

                        Please notice that this running version is trying to keep punctuation marks in the wikipedia
                        articles. for this we will be using MyWikiCorpus.py copied from the user rhazegh from:
                        https://github.com/RaRe-Technologies/gensim/issues/552#issuecomment-278036501.

                        To use the normal Wikicorpus just switch the comments on the instance of 'wiki'.

                        To use WikiCorpus_inst.pickle, comment out both wiki instantiation and the pickle.dump
                        and remove the comment on the instance of wiki with pickle.load

                        Wikipedia dump download links can be found here: https://dumps.wikimedia.org/enwiki/latest/
                        example name for download purposes: enwiki-latest-pages-articles.xml.bz2

    :param wiki_dump:   A file of a wikipedia dump.
    :param file_name:   The name of the created file, default is set to wiki_corpus.txt.

    :return:            Creates a txt file containing each wikipedia article in a separate line : wiki_corpus.txt.
    :return:            Dumps a pickle file with the WikiCorpus instance : WikiCorpus_inst.pickle.
    :return:            Dumps the number of articles in the wikipedia corpus created : wiki_corpus_length.pickle.
    """

    sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace')
    if sys.stdout.encoding != 'cp850':
        sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding != 'cp850':
        sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')
    output = open(file_name, 'wb')
    print("creating wiki_corpus inst")

    # wiki = WikiCorpus(wiki_dump, article_min_tokens=100, lemmatize=False)
    wiki = MyWikiCorpus(wiki_dump)
    pickle.dump(wiki, open('WikiCorpus_inst.pickle', 'wb'))
    # wiki = pickle.load(open('WikiCorpus_inst.pickle', "rb"))
    print("finished creating wiki_corpus inst")
    text_num = pickle.load(open('wiki_corpus_length.pickle', "rb"))
    print("Total Wikipedia Articles: " + str(text_num))
    titl = 'Creating ' + file_name
    bar = pyprind.ProgBar(text_num, title=titl, stream=sys.stdout, width=90)
    for text in wiki.get_texts():
        bar.update()
        text = ' '.join(text).encode('utf-8') + '\n'.encode('utf-8')
        # output.write((' '.join(text), 'utf-8').decode('utf-8') + '\n')
        output.write(text)
    output.close()
    print('Processing complete!')


# the number of words in each line printed is set by print_param.
def check_corpus(corpus_list_of_lists_of_str, new_line_param=40):
    """
                This function is used to print out the articles from the wikipedia dump.
                If you wish to stop reading the articles please

    :param corpus_list_of_lists_of_str: A corpus in the form of list of lists of str.
    :param new_line_param:                 The number of words in each line printed.

    :return:                            Prints the articles from the list.
    """
    user_input = 'FIRST'
    list_num = 0
    while user_input != 'STOP':
        full_line_to_print = corpus_list_of_lists_of_str[list_num]
        line_to_print = ''
        word_counter = 0
        for word in full_line_to_print:
            line_to_print = line_to_print + word + ' '
            word_counter += 1
            if word_counter % new_line_param == 0:
                print(line_to_print)
                line_to_print = ''
        if word_counter % new_line_param != 0:
            print(line_to_print)
        user_input = input('>>> Type \'STOP\' to quit or hit Enter key for more <<< ')
        if user_input == 'STOP':
            break
        list_num += 1


def create_corpus_from_wiki_endoded_text(input_file):
    """
                                This function is used to get change the txt file into a list of lists of str so our
                                word2vec and doc2vec models will be able to use them.

                                Uses wiki_corpus_length.pickle created by the function make_corpus.

    :param      input_file:     A wikipedia corpus txt file: wiki_corpus.txt, created by the function make_corpus.

    :return:    wiki_corpus:    A list of lists of str.
    :return:                    Dumps a pickle file with the list of lists of str :
                                wiki_corpus_list_of_lists_of_str.pickle.

    """
    wiki_corpus = []
    text_num = pickle.load(open('wiki_corpus_length.pickle', "rb"))
    bar = pyprind.ProgBar(text_num, title='Creating wiki_corpus_full.pickle', stream=sys.stdout, width=100)
    i = 0
    while i != text_num:
        bar.update()
        i += 1
        line = input_file.readline()
        if not line:
            break
        line_d = line.decode('utf-8')
        article_list_of_str = line_d.split()
        wiki_corpus.append(article_list_of_str)
    print("starting to save file")
    pickle.dump(wiki_corpus, open('wiki_corpus_list_of_lists_of_str.pickle', 'wb'))
    print("finished to save file")
    return wiki_corpus


def load_corpus(input_file):
    """Loads corpus from text file"""

    print('Loading corpus...')
    time1 = time.time()
    corpus = input_file.read()
    time2 = time.time()
    total_time = time2 - time1
    print('It took %0.3f seconds to load corpus' % total_time)
    return corpus


# text_as_words = [[w1, w2, w3], [u1, u2, u3, u4, ...], ...]
# text_as_sentences = [w1 w2 w3, u1 u2 u3 u4..., ...]
def words2sentences(text_as_words):
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    for sent in text_as_words:
        full_sent = ''
        for w in sent:
            string = tokenizer.tokenize(w)[0] + ' '
            string_without_numbers = re.sub(r'\d+', '', string)
            full_sent += string_without_numbers
        sentences.append(full_sent)
    return sentences


def sentences2words(text_as_sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    for sent in text_as_sentences:
        assert isinstance(sent, str)
        sent = re.sub(r'\d+', '', sent)
        sentences.append(tokenizer.tokenize(sent))
    return sentences


def file2sentences(file_name):
    f = open(file_name, "r")
    text = f.read()
    text_without_numbers = re.sub(r'\d+', '', text)
    return sent_tokenize(text_without_numbers)


def remove_stopwords_from_words(words):
    new_words = []
    for words_list in words:
        new_words_list = []
        for word in words_list:
            word = word.lower()
            if word not in stopwords.words('english'):
                new_words_list.append(word)
        new_words.append(new_words_list)
    return new_words


def remove_by_pos(text_as_words, remove):
    tagged_text_as_words = pos_tag_sents(text_as_words)
    new_text_as_words = []
    for sentences in tagged_text_as_words:
        new_sentence = []
        for w, tag in sentences:
            if len(w) > 1 and tag not in remove:
                new_sentence.append(w)
        new_text_as_words.append(new_sentence)
    return new_text_as_words


def text_to_list_of_lists(text_name):
    sentences = file2sentences(text_name)
    words = sentences2words(sentences)
    return words


# if __name__ == '__main__':
#     sentences = file2sentences('wikipedia-mouse.txt')
#     print(sentences)
#     words = sentences2words(sentences)
#     print(words, sep="\n")


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
    #     sys.exit(1)
    in_f = "enwiki-latest-pages-articles.xml.bz2"
    make_corpus(in_f)
    corpus_file = open("wiki_corpus.txt", 'rb')
    # check_corpus(corpus_file)
    wiki_corpus_list_of_articals_of_str = create_corpus_from_wiki_endoded_text(corpus_file)
