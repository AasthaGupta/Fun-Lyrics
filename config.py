# -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-19 08:23:18
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-04-25 14:22:54

import pickle
import os


#change artist name and url
ARTIST_NAME = "rihanna"
URL = "http://www.metrolyrics.com/rihanna-lyrics.html"

RESOURCE_PATH = "resources/"
PATH = os.path.join(RESOURCE_PATH,ARTIST_NAME)

# make directory for artist if doesn't exist
if not os.path.exists(PATH):
    os.makedirs(PATH)

LYRICS_FILE = os.path.join(PATH,"lyrics.txt")
DATA_FILE = os.path.join(PATH,"dataset.txt")
MODEL_FILE = os.path.join(PATH,"model.h5")
OUTPUT_FILE = os.path.join(PATH,"fun.txt")

# directory of checkpoints
CHKPT_PATH = os.path.join(PATH,"checkpoints/")
if not os.path.exists(CHKPT_PATH):
    os.makedirs(CHKPT_PATH)

# get this data from file if exists
pkl_filename = os.path.join(PATH,"config_data.pkl")
if os.path.exists(pkl_filename):
	with open(pkl_filename,"rb") as f:
		VOCAB_SIZE, DATA_SIZE, NUM_SEQ  = pickle.load(f)
else:
	VOCAB_SIZE = 0
	DATA_SIZE = 0
	NUM_SEQ = 0 # number of sequences = DATA_SIZE/SEQ_LENGTH

SEED = 7

SEQ_LENGTH = 50

# number of units of LSTM
HIDDEN_DIM = 256
# number of layers in the network
LAYER_NUM = 3
# number of epochs for training
NUM_EPOCHS = 250
# batch size
BATCH_SIZE = 100
# Initial learning rate
LR = 0.0001

# length of characters to generate
LEN_TO_GEN = 700

# length of characters to generate(for analysis.py)
LEN_TO_GEN_2 = 250