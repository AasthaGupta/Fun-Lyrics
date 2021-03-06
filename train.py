# -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-18 02:08:44
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-05-19 05:46:00

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback, LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
import keras.backend as K
import preprocess
import os
import config
if config.FLOYD == True:
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# fix random seed for reproducibility
numpy.random.seed(config.SEED)

# define the LSTM model
def create_model():

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

	# define optimiser for the model
	optimiser = Adam(lr=config.LR)

	# compile this model
	model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=['accuracy'])

	return model


# to print learning rate
class LearningRatePrinter(Callback):
	def on_epoch_begin(self, epoch, logs={}):
		optimizer = self.model.optimizer
		initial_lr = K.eval(optimizer.lr)
		print ( "learning rate: {:.6f}".format(initial_lr) )
		# decay = K.eval(optimizer.decay)
		# iterations = K.eval(optimizer.iterations)
		# lr =  initial_lr * (1. / (1. + decay * iterations))
		# print ( "initial_lr: {:.6f}".format(initial_lr),
		# 	    "lr: {:.6f}".format(lr),
		# 	    "decay: {:.6f}".format(decay),
		# 	    "iteration: {}".format(iterations) )

def train():

	# to change learnig rate every 100 interations
	# def Scheduler(epoch):
	# 	lr = K.eval(model.optimizer.lr)
	# 	if epoch == 10:
	# 		new_lr = lr * 0.1
	# 	# elif epoch == 12:
	# 	# 	new_lr = 0.0001
	# 	# elif epoch == 25:
	# 	# 	new_lr = 0.002
	# 	elif epoch != 0 and epoch % 30 == 0:
	# 		new_lr = lr * 0.1
	# 	else:
	# 		new_lr = lr
	# 	model.optimizer.lr.assign(new_lr)
	# 	return new_lr

	X,Y = preprocess.build_dataset()
	print('X shape:', X.shape)
	print('Y shape:', Y.shape)

	temp_model_file = os.path.join(config.PATH,"temp_model.h5")
	if os.path.exists(temp_model_file):
		model = load_model(temp_model_file)
		print ("LSTM Network loaded")
	else:
		model = create_model()
		print ("LSTM Network created")

	# model summary
	print ("Model Summary:")
	print (model.summary())

	# define the checkpoint and learning rate change
	filepath = os.path.join(config.CHKPT_PATH,"weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5")
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, min_lr=0.000001)
	logfilepath=os.path.join(config.PATH,"logs.csv")
	logger = CSVLogger(logfilepath)
	stopper = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=15, verbose=1, mode='auto')
	# lr_change = LearningRateScheduler(Scheduler)
	callbacks_list = [checkpoint, reduce_lr, logger, LearningRatePrinter(), stopper]

	# fit the model
	history = model.fit(X, Y,
						batch_size=config.BATCH_SIZE,
						validation_split=0.2,
						verbose=2,
						epochs=config.NUM_EPOCHS,
						callbacks=callbacks_list)
	print("LSTM Network trained")

	# save model
	model.save(config.MODEL_FILE)
	print("LSTM Network saved")

	# serialize model to JSON
	model_json = model.to_json()
	json_filename = os.path.join(config.PATH,"model_json.json")
	with open(json_filename, "w") as json_file:
		json_file.write(model_json)

	# delete the existing model
	del model

	# save history
	pkl_filename = os.path.join(config.PATH,"history.pkl")
	pickle.dump(history.history, open(pkl_filename, "wb"))

	# statistics of training the model
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('LSTM accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt_filename = os.path.join(config.PATH,'LSTM accuracy.png')
	plt.savefig(plt_filename)
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('LSTM loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt_filename = os.path.join(config.PATH,'LSTM loss.png')
	plt.savefig(plt_filename)
	plt.show()

if __name__ == "__main__":
	train()