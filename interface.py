from word_spliter import *
from Text_Processing import text_to_list_of_lists
import pickle
from termcolor import colored
# pickle.dump(model_accuracy, open('accuracy for ' + INFERENCE_FILE_NAME + TRAIN_FILE_NAME + W_FILE, 'wb'))
# model_accuracy = pickle.load(open('accuracy for ' + INFERENCE_FILE_NAME + TRAIN_FILE_NAME + W_FILE, "rb"))


if __name__ == '__main__':
    brown_corpus = brown.sents()
    webtext_corpus = webtext.sents()
    wikipedia_mouse_corpus = text_to_list_of_lists('wikipedia-mouse.txt')
    from nltk.corpus import gutenberg
    books = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt',
             'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt',
             'chesterton-brown.txt','chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
             'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt','shakespeare-macbeth.txt',
             'whitman-leaves.txt']
    gutenberg_corpus = []
    for book in books:
        gutenberg_corpus += list(gutenberg.sents(book))
    print(len(gutenberg_corpus))

    mega_corpus = []
    for sent in brown_corpus:
        mega_corpus += list(sent)
    for sent in webtext_corpus:
        mega_corpus += list(sent)
    for sent in wikipedia_mouse_corpus:
        mega_corpus += list(sent)
    for sent in gutenberg_corpus:
        mega_corpus += list(sent)

    print("Hello,\n"
          "Welcome to the Word Splitting Program.\n",
          colored('Please type the number of the answer.', 'blue'))
    # print("Would you like to update the corpora and the english language stop-words?")
    answer_1 = 0    # answer_1 = int(input("1. Yes\t2. No\n"))
    print("Which corpus would you like to use?")
    answer_3 = int(input("1. Wikipedia-mouse-testing\t2. Webtext\t3. Brown\t4. Gutenberg\t5. All\n"))
    print("Would you like to like to create a new word_splitter instance?")
    answer_2 = int(input("1. Yes\t2. No\n"))
    print("Would you like to check for similar words from the new language corpus?")
    answer_5 = int(input("1. Yes\t2. No\n"))

    if answer_1 == 1:
        import nltk
        nltk.download()

    if answer_3 == 1:
        using_corpus = '_wikipedia_mouse'
        corpus_ = wikipedia_mouse_corpus
    elif answer_3 == 2:
        using_corpus = '_webtext'
        corpus_ = webtext_corpus
    elif answer_3 == 3:
        using_corpus = '_brown'
        corpus_ = brown_corpus
    elif answer_3 == 4:
        using_corpus = '_gutenberg'
        corpus_ = gutenberg_corpus
    elif answer_3 == 5:
        using_corpus = '_all'
        corpus_ = mega_corpus

    if answer_2 == 1:
        word_splitter = WordSplitter(corpus=corpus_)
        pickle.dump(word_splitter, open('word_splitter_instance' + using_corpus, 'wb'))
        print("#### word_splitter_instance Saved Successfully ####")
    else:
        word_splitter = pickle.load(open('word_splitter_instance' + using_corpus, "rb"))
        print("#### word_splitter_instance Loaded Successfully ####")

    old_word2vec_model = word_splitter.word2vec_model
    new_corpus = word_splitter.create_new_language_corpus()
    print("#### new_language_corpus Loaded Successfully ####")
    new_word2vec_model = word_splitter.create_new_language_w2v()
    print("#### new_language_word2vec_model Loaded Successfully ####")
    print_dictionary_to_excel(word_splitter, using_corpus)
    print_canonic_words_to_excel(word_splitter, using_corpus)

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
