from word_spliter import *
import pickle
# pickle.dump(model_accuracy, open('accuracy for ' + INFERENCE_FILE_NAME + TRAIN_FILE_NAME + W_FILE, 'wb'))
# model_accuracy = pickle.load(open('accuracy for ' + INFERENCE_FILE_NAME + TRAIN_FILE_NAME + W_FILE, "rb"))

if __name__ == '__main__':
    print("Hello,\n"
          "Welcome to the Word Splitting Program.")
    print("Would you like to update the Brown corpus and the english language stop-words?\n"
          "Type the # of the answer:")
    answer_1 = int(input("1. Yes\n2. No\n"))
    print(answer_1)
    print("Would you like to like to create a new word_splitter instance?\n"
          "Type the # of the answer:")
    answer_2 = int(input("1. Yes\n2. No\n"))
    print(answer_2)
    print("Would you like to like to create a new language?\n"
          "Type the # of the answer:")
    answer_3 = int(input("1. Yes\n2. No\n"))
    print(answer_3)
    print("Would you like to like to create a new Word2Vec model for the new language corpus?\n"
          "Type the # of the answer:")
    answer_4 = int(input("1. Yes\n2. No\n"))
    print(answer_4)
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
        new_corpus = word_splitter.classify(word_splitter.corpus)
        pickle.dump(new_corpus, open('new_language_corpus', 'wb'))
        print("#### new_language_corpus Saved Successfully ####")
    else:
        new_corpus = pickle.load(open('new_language_corpus', "rb"))
        print("#### new_language_corpus Loaded Successfully ####")

    if answer_4 == 1:
        new_word2vec_model = word2vec_feat(new_corpus)
        pickle.dump(new_word2vec_model, open('new_language_word2vec_model', 'wb'))
        print("#### new_language_word2vec_model Saved Successfully ####")
    else:
        new_word2vec_model = pickle.load(open('new_language_word2vec_model', "rb"))
        print("#### new_language_word2vec_model Loaded Successfully ####")
    # print(new_corpus)
    # print('Plotting:')
    # test_word = "computer"
    # plot_close_words(old_word2vec_model, test_word)
    # plot_close_words(new_word2vec_model, test_word)
