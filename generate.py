# -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-18 12:13:33
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-05-19 05:47:30

import numpy as np
from keras.models import load_model
import pickle
import config
import os
import string
import argparse

model = None
int_to_char = None
char_to_int = None


def load_saved_model():
	global model
	model = load_model(config.MODEL_FILE)


def load_mapping():
	global char_to_int
	global int_to_char

	pkl_filename = os.path.join(config.PATH,"dict.pkl")
	with open(pkl_filename,"rb") as f:
		char_to_int = pickle.load(f)

	pkl_filename = os.path.join(config.PATH,"rev_dict.pkl")
	with open(pkl_filename,"rb") as f:
		int_to_char = pickle.load(f)


def generate(seed_char):

	global model
	global char_to_int
	global int_to_char

	text = seed_char
	seed = char_to_int[seed_char]

	X = np.zeros((1, config.LEN_TO_GEN, config.VOCAB_SIZE))
	i = 0

	while i < config.LEN_TO_GEN:
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



def generate_text():
	global model
	global char_to_int
	global int_to_char

	parser = argparse.ArgumentParser()
	parser.add_argument("-s","--seed", type=str, help="seed to generate text", default="life")

	load_saved_model()
	print ("Loaded saved LSTM Network")
	load_mapping()
	print ("Loaded saved character mappings")

	print ("Generating Lyrics....")

	seed = parser.parse_args().seed
	print (len(seed))
	output = seed

	padding = ""
	for i in range(config.SEQ_LENGTH-len(seed)):
		padding = padding + " "


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



	# output = ""
	# length = 0
	# alpha_dict = string.ascii_lowercase
	# while length < config.LEN_TO_GEN:
	# 	# pick a random seed that is alphabet
	# 	seed_char = alpha_dict[np.random.randint(23)]
	# 	text,i = generate(seed_char)
	# 	length = length + i
	# 	output = output + "\n".join(s.capitalize() for s in text.split("\n"))

	# format output
	output = "\n".join(s.capitalize() for s in output.split("\n"))

	# print output
	print(output)

	# save generated lyrics in output file
	output_filename = config.OUTPUT_FILE
	with open(output_filename,"w") as f:
		f.write(output)
	print( "Saved in fun.txt file! :)" )

if __name__ == "__main__":
    generate_text()