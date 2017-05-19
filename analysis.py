 # -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-21 03:19:48
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-05-19 15:51:13

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
# define model's network here
model.add(LSTM(config.HIDDEN_DIM, input_shape=(config.SEQ_LENGTH, config.VOCAB_SIZE), return_sequences=True))
model.add(Dropout(0.2))
for i in range(config.LAYER_NUM - 1):
	if i != config.LAYER_NUM - 2:
		model.add(LSTM(config.HIDDEN_DIM, return_sequences=True))
	else:
		model.add(LSTM(config.HIDDEN_DIM))
model.add(Dense(config.VOCAB_SIZE, activation = "softmax"))

# open file handle
f = open(config.ANALYSIS_FILE,"w")

seed = "life"
print (len(seed))
output = seed

padding = ""
for i in range(config.SEQ_LENGTH-len(seed)):
	padding = padding + " "

# load the network weights
filepath = config.CHKPT_PATH
for filename in glob.glob(os.path.join(filepath, '*.hdf5')):
	print(os.path.basename(filename)[:-5])
	f.write(os.path.basename(filename)[:-5]+"\n")
	model.load_weights(filename)
	# compile this model
	model.compile(loss="categorical_crossentropy", metrics=['accuracy'])

	X_sequence = padding + seed
	X_sequence_int = [char_to_int[c] for c in X_sequence]


	for i in range(config.LEN_TO_GEN):
		input_sequence = np.zeros((1, config.SEQ_LENGTH, config.VOCAB_SIZE))
		for j in range(config.SEQ_LENGTH):
			input_sequence[0][j][X_sequence_int[j]] = 1.
		Y = model.predict(input_sequence)
		next_int = np.argmax(Y, 1)[-1]
		next_char = int_to_char[next_int]
		# print(next_int,next_char)
		output = output + next_char
		del(X_sequence_int[0])
		X_sequence_int.append(next_int)

	# format output
	output = "\n".join(s.capitalize() for s in output.split("\n"))

	# print output
	print(output)

	# save generated lyrics in output file
	output_filename = config.OUTPUT_FILE
	with open(output_filename,"w") as f:
		f.write(output)
	print( "Saved in fun.txt file! :)" )

# close file handle
f.close()
print( "Saved in analysis.txt file!" )
