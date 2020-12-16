# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:23:05 2020

@author: PRATIK
"""
# Install tf-nightly so that the tf.lite interpretor works correctly


from Functional_FLow import final_fun_1

user_queston = input('Enter question : ')
answer = final_fun_1(user_queston)
print("Predicted Answer is : ", answer)
