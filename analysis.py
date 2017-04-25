 # -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-21 03:19:48
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-04-25 14:26:42

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
import config
import os
import glob
import pickle
import string

def generate(seed_char):

	global model
	global char_to_int
	global int_to_char

	text = seed_char
	seed = char_to_int[seed_char]

	X = np.zeros((1, config.LEN_TO_GEN_2, config.VOCAB_SIZE))
	i = 0

	while i < config.LEN_TO_GEN_2:
		X[0, i, :][seed] = 1.
		Y = model.predict(X[:, :i+1, :])[0]
		next = np.argmax(Y, 1)
		seed = next[-1]
		next_char = int_to_char[seed]
		text = text + next_char
		i = i + 1
		# print (int_to_char[seed],seed)
		# print (X[0, i, seed])
		# print (Y)
		# print(next)
		# print (int_to_char[seed],seed)
		# if next_char == '\n':
		# 	break

	return text,i

# load data
pkl_filename = os.path.join(config.PATH,"dict.pkl")
with open(pkl_filename,"rb") as f:
	char_to_int = pickle.load(f)

pkl_filename = os.path.join(config.PATH,"rev_dict.pkl")
with open(pkl_filename,"rb") as f:
	int_to_char = pickle.load(f)

# define the LSTM model
model = Sequential()
model.add(LSTM(config.HIDDEN_DIM, input_shape=(None, config.VOCAB_SIZE), return_sequences=True))
model.add(Dropout(0.2))
for i in range(config.LAYER_NUM - 1):
    model.add(LSTM(config.HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(config.VOCAB_SIZE)))
model.add(Activation('softmax'))

# load the network weights
filepath = config.CHKPT_PATH
for filename in glob.glob(os.path.join(filepath, '*.hdf5')):
	print(os.path.basename(filename)[:-5])
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	output = ""
	length = 0
	alpha_dict = string.ascii_letters
	while length < config.LEN_TO_GEN_2:
		seed_char = alpha_dict[np.random.randint(45)]
		text,i = generate(seed_char)
		length = length + i
		output = output + text

	print(output)
