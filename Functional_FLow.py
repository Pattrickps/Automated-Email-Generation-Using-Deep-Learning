# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:02:31 2020

@author: PRATIK
"""

from Functional_Library import load_tokenizer, one_hot_decode, str_to_tokens, preprocess_usr
import numpy as np
#from numpy import array
import tensorflow as tf
print(tf.__version__)
import warnings
warnings.filterwarnings("ignore")

#Code References : https://github.com/bhattbhavesh91/tflite-tutorials/blob/master/tflite-part-1.ipynb
#https://github.com/bhattbhavesh91/tflite-tutorials/blob/master/tflite-part-2.ipynb


# This function will take the raw user input (Question) as the 1st input.
# Additionally, we will also declare the Number of words (n_steps) we want in the output. 
# But if the model predicts 'samaapt' before completing n_steps, then the model will stop prediction then and there
# Return value: This function will return the predicted answer in English language

n_steps = 50 # Define n_steps as Global variable


def final_fun_1(user_queston):
    # First, pre-process the user question
    user_queston = preprocess_usr(user_queston)

    # Append the user question into the question list
    #question_list.append([user_queston])

    # Initialize the interpreter for the TFLite Inference Encoder model
    TF_LITE_MODEL_FILE_NAME_1 = "Model_14/tf_lite_infenc.tflite"
    interpreter_1 = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME_1)
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()

    # Initialize the interpreter for the TFLite Inference Decoder model
    TF_LITE_MODEL_FILE_NAME_2 = "Model_14/tf_lite_infdec.tflite"
    interpreter_2 = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME_2)
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    # Encoder Input
    li = str_to_tokens( user_queston )

    # Changing the input to float32 bcoz the encoder input expects the type to be float32
    li2= np.array(li, dtype=np.float32)

    # Allocate tensor, set tensor, invoke, get tensor
    # This interpreter requires only 1 input of shape [1,154] and fetches 3 Outputs/Predictions
    interpreter_1.allocate_tensors()
    interpreter_1.set_tensor(input_details_1[0]['index'], li2)

    interpreter_1.invoke()

    # Encoder Predictions
    # The first prediction is of shape (1,154,100)
    tflite_model_predictions_1_0 = interpreter_1.get_tensor(output_details_1[0]['index'])

    # The second prediction is of shape (1,100)
    tflite_model_predictions_1_1 = interpreter_1.get_tensor(output_details_1[1]['index'])

    # The third prediction is of shape (1,100)
    tflite_model_predictions_1_2 = interpreter_1.get_tensor(output_details_1[2]['index'])

    output = list()

    # Load the tokenizer
    tokenizer = load_tokenizer()
    # target_seq will be the 3rd input to the decoder
    target_seq = np.zeros( ( 1 , 1 ), dtype=np.float32 ) 
    target_seq[0, 0] = tokenizer.word_index['prarambh']

    for _ in range(n_steps):
        # Allocate(only once), [set, invoke , get]
        # First Input for the Decoder has to be shape of (1,100)
        interpreter_2.allocate_tensors()
        interpreter_2.set_tensor(input_details_2[0]['index'], tflite_model_predictions_1_1)

        # Second Input Decoder has to be shape of (1,100)
        interpreter_2.set_tensor(input_details_2[1]['index'], tflite_model_predictions_1_2)

        # Third input Decoder has to be shape of (1,1)
        interpreter_2.set_tensor(input_details_2[2]['index'], target_seq)

        # Fourth input Decoder has to be shape of (1,154,100) and this input will remain constant for all the iteration of the for loop
        interpreter_2.set_tensor(input_details_2[3]['index'], tflite_model_predictions_1_0)
        interpreter_2.invoke()

        # Decoder Predictions- We only need the 1st prediction(1,100 ) , 2nd prediction (1,1,VOCAB_SIZE), 3rd prediction (1,100). 
        # 4th prediction (attention weights) are not useful
        # tflite_model_predictions_2_0 has shape of (1,100 )
        tflite_model_predictions_2_0 = interpreter_2.get_tensor(output_details_2[0]['index'])

        # tflite_model_predictions_2_1 has shape of (1,1,VOCAB_SIZE)
        tflite_model_predictions_2_1 = interpreter_2.get_tensor(output_details_2[1]['index'])

        # tflite_model_predictions_2_2 has shape of (1,100 )
        tflite_model_predictions_2_2 = interpreter_2.get_tensor(output_details_2[2]['index'])

        output.append(tflite_model_predictions_2_1[0, 0, :])
        # Getting data ready for next iteration
        index = np.argmax( tflite_model_predictions_2_1[0, 0, :] )
        if index != tokenizer.word_index['samaapt']: # At a given iteration if the prediction is 'end', then the loop should break
            target_seq = np.zeros( ( 1 , 1 ), dtype=np.float32 )  
            target_seq[ 0 , 0 ] = index
        else:
            break
        
        tflite_model_predictions_1_1 = tflite_model_predictions_2_0
        tflite_model_predictions_1_2 = tflite_model_predictions_2_2

    answer_out = one_hot_decode(np.array(output))

    # Append the model's predicted answer in the answer list
    #answer_list.append([answer_out])

    return answer_out
