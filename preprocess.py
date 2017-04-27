# -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-19 08:21:36
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-04-27 23:59:57

import config
import string
import numpy as np
import pickle
import os
import re


def build_dataset():

	# only allow letters, digits, space, newline and some punctuations
	punc_chars = "\n " + "!',?"
	allowed_chars = string.ascii_letters + punc_chars + string.digits

	# remove all other letters from raw text and save the data
	# if already done it then open that file
	if os.path.exists(config.DATA_FILE):
		with open(config.DATA_FILE,"r") as f:
			data = f.read()
	else:
		data = ""
		# all the raw data fetchd from the website
		with open(config.LYRICS_FILE,"r") as f:
			for line in f.readlines():

				# remove anything within [] and ()
				line = re.sub("[\(\[].*?[\)\]]", "", line)
				if not line:
					continue

				# keep only allowed chars
				line = ''.join([char for char in line if char in allowed_chars])
				if not line:
					continue

				# if this line just says "Chorus"
				if line.strip() == "Chorus" :
					continue

				# line shouldn't start with punctuations or space or newline
				while line and line[0] in punc_chars:
					line = line[1:]
				if not line:
					continue

				# remove trailing whitespaces
				line = line.strip()
				# continue if blank line
				if not line:
					continue

				data += line.lower() + "\n"

		with open(config.DATA_FILE,'w') as f:
			f.write(data)

	s=0
	c=0
	for line in data.split("\n"):
		l=len(line.strip())
		# print(l)
		s+=l
		c+=1
	avg = s//c
	print(avg)


	chars = sorted(list(set(data)))
	# map integer to char and a reverse lookup table
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))

	pkl_filename = os.path.join(config.PATH,"dict.pkl")
	pickle.dump(char_to_int, open(pkl_filename, "wb"))
	pkl_filename = os.path.join(config.PATH,"rev_dict.pkl")
	pickle.dump(int_to_char, open(pkl_filename, "wb"))

	config.VOCAB_SIZE = len(chars)
	config.DATA_SIZE = len(data)
	config.NUM_SEQ = config.DATA_SIZE // config.SEQ_LENGTH

	pkl_filename = os.path.join(config.PATH,"config_data.pkl")
	pickle.dump([config.VOCAB_SIZE, config.DATA_SIZE , config.NUM_SEQ ], open(pkl_filename, "wb"))

	print( "Vocabulary size:", config.VOCAB_SIZE)
	print( "Data size:", config.DATA_SIZE)
	print( "Length of Sequence:", config.SEQ_LENGTH)
	print( "Number of Sequences:", config.NUM_SEQ)

	# make input and output pairs from the data
	X = np.zeros((config.NUM_SEQ, config.SEQ_LENGTH, config.VOCAB_SIZE))
	Y = np.zeros((config.NUM_SEQ, config.VOCAB_SIZE))
	for i in range(config.NUM_SEQ):
		X_sequence = data[i*config.SEQ_LENGTH : (i+1)*config.SEQ_LENGTH]
		X_sequence_int = [char_to_int[c] for c in X_sequence]
		input_sequence = np.zeros((config.SEQ_LENGTH, config.VOCAB_SIZE))
		for j in range(config.SEQ_LENGTH):
			input_sequence[j][X_sequence_int[j]] = 1.
		X[i] = input_sequence

		Y_sequence = data[(i+1)*config.SEQ_LENGTH]
		Y_sequence_int = [char_to_int[c] for c in Y_sequence]
		target_sequence = np.zeros((config.VOCAB_SIZE))
		target_sequence[Y_sequence_int[0]] = 1.
		Y[i] = target_sequence

	# print(X,Y)
	return X,Y

if __name__ == "__main__":
    build_dataset()