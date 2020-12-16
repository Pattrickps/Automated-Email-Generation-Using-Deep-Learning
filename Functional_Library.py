# -*- coding: utf-8 -*-
import pickle
import re
import tensorflow as tf
from numpy import argmax
####################################################################
             #  Helper Function 1
"""
1.Let's first load the saved tokenizer object, so that we can directly use it in Evaluation data
"""
####################################################################

def load_tokenizer():
    # loading the  saved tokenizer - ref: https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
    tokenizer_path = 'Model_14/tokenizer.pickle'
    with open(tokenizer_path, 'rb') as obj:
        tokenizer = pickle.load(obj)
    return tokenizer
        
# 2.Let's write 3 helper the functions that will be called in the final function which will predict the answer to the question

####################################################################
             #  Helper Function 2
"""
This function will be called inside the predict_answer function.
The parameter of this function is of the size : (Number of Output steps , VOCAB_SIZE)
"""
####################################################################


def one_hot_decode(encoded_seq):
    # li will be a list like [12,2230,2345......Number of Output steps times]
    li = [argmax(vector) for vector in encoded_seq]
    decoded_translation = ''
    # Load the tokenizer
    tokenizer = load_tokenizer()
    for value in li:
        for word , index in tokenizer.word_index.items() :
            if value == index:
                decoded_translation += ' {}'.format( word )

    return decoded_translation


####################################################################
            # Helper Function 3
"""
This function will convert the pre-processed User question into a sequence of tokenizer tokens.
"""
####################################################################


def str_to_tokens( sentence : str ):
    # Load the tokenizer
    tokenizer = load_tokenizer()    
    maxlen_questions = 154 # We get this value while creating the data for training.
    # This will also take care of out of vocabulary tokens; OOV tokens will be simply ignored in the Sequence
    tokens = tokenizer.texts_to_sequences([sentence])
    return tf.keras.preprocessing.sequence.pad_sequences( tokens , maxlen=maxlen_questions , padding='post')


####################################################################
            # Helper Function 4
"""
tokenizer.texts_to_sequences(["in cleaning up some template, information"]) and tokenizer.texts_to_sequences(["in cleaning up some template information"])
will give the same output because the tokenizer simply filters out the ","
But if its written tokenizer.texts_to_sequences(["in cleaning up some template, information."])
Then it will not filter the "." and also it will not tokenize "information."
Therefore, always pass the input sentence through a processor that will put a Space between "information" and "."
and tokenizer will assign token to both "information" and "."
Same reasoning goes for "?" as well
"""
####################################################################


# Replace apostrophe/short words in python-  https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python

def preprocess_usr(sentence):
    # replace multiple spaces with single space
    sentence = re.sub('\s+',' ', sentence)
    #This is to be done because I want to Include punctuation and Question mark in keras tokenizer.
    #I do not want the Tokeniozer API to remove them
    sentence = sentence.replace(".", " .")
    sentence = sentence.replace(",", " ,")
    sentence = sentence.replace("?", " ?")
    # specific
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    # converting all the chars into lower-case.
    sentence = sentence.lower()

    return sentence