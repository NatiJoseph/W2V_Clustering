from word_spliter import *
import pickle
from termcolor import colored
# pickle.dump(model_accuracy, open('accuracy for ' + INFERENCE_FILE_NAME + TRAIN_FILE_NAME + W_FILE, 'wb'))
# model_accuracy = pickle.load(open('accuracy for ' + INFERENCE_FILE_NAME + TRAIN_FILE_NAME + W_FILE, "rb"))

if __name__ == '__main__':
    print("Hello,\n"
          "Welcome to the Word Splitting Program.\n",
          colored('Please type the number of the answer.', 'blue'))
    print("Would you like to update the Brown corpus and the english language stop-words?")
    answer_1 = int(input("1. Yes\t2. No\n"))
    print("Would you like to like to create a new word_splitter instance?")
    answer_2 = int(input("1. Yes\t2. No\n"))
    print("Would you like to like to create a new language?")
    answer_3 = int(input("1. Yes\t2. No\n"))
    print("Would you like to like to create a new Word2Vec model for the new language corpus?")
    answer_4 = int(input("1. Yes\t2. No\n"))
    print("Would you like to check for similar words from the new language corpus?")
    answer_5 = int(input("1. Yes\t2. No\n"))
    if answer_1 == 1:
        nltk.download('brown')
        nltk.download('stopwords')
    # from nltk.corpus import brown
    # print('Start')
    # print(stopWords)
    # print(brown_corpus)
    brown_corpus = brown.sents()
    if answer_2 == 1:
        word_splitter = WordSplitter(corpus=brown_corpus)
        pickle.dump(word_splitter, open('word_splitter_instance', 'wb'))
        print("#### word_splitter_instance Saved Successfully ####")
    else:
        word_splitter = pickle.load(open('word_splitter_instance', "rb"))
        print("#### word_splitter_instance Loaded Successfully ####")
    old_word2vec_model = word_splitter.word2vec_model
    if answer_3 == 1:
        new_corpus = word_splitter.create_new_language_corpus()
        pickle.dump(new_corpus, open('new_language_corpus', 'wb'))
        print("#### new_language_corpus Saved Successfully ####")
    else:
        new_corpus = pickle.load(open('new_language_corpus', "rb"))
        print("#### new_language_corpus Loaded Successfully ####")

    if answer_4 == 1:
        new_word2vec_model = word_splitter.create_new_language_w2v()
        pickle.dump(new_word2vec_model, open('new_language_word2vec_model', 'wb'))
        print("#### new_language_word2vec_model Saved Successfully ####")
    else:
        new_word2vec_model = pickle.load(open('new_language_word2vec_model', "rb"))
        print("#### new_language_word2vec_model Loaded Successfully ####")

    while answer_5 == 1:
        answer_word = str(input("Enter word:\t")).lower()
        answer_word_new_words = word_splitter.get_new_words(answer_word)
        print("The new words and their most similar word in the new language:")
        print(answer_word_new_words, sep="\n")
        print("Would you like to check another word?")
        answer_5 = int(input("1. Yes\t2. No\n"))

    # if answer_5 == 1:
    #     print(new_corpus, sep='\n')
    #     print(len(new_corpus))

    # print(new_corpus)
    # print('Plotting:')
    # test_word = "computer"
    # plot_close_words(old_word2vec_model, test_word)
    # plot_close_words(new_word2vec_model, test_word)
