#! /usr/bin/python3

import numpy as np

from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Normalization
from tensorflow.keras.optimizers import Adam

train_file = r"/home/dan/Documents/SentimentAnal/train.csv"
test_file = r"/home/dan/Documents/SentimentAnal/test.csv"

def unpack_data(filename):
	results = []
	with open(filename, 'rb') as file:
		for i, line in enumerate(file):
			line_string = str(line)
			message = ''
			elements = line_string.split(',')
			for j, element in enumerate(elements):
				if j == 0:
					continue
				if element == 'neutral':
					message = message.strip()
					data = [message, 0]
					results.append(data)
					break
				elif element == 'positive':
					message = message.strip()
					data = [message, 1]
					results.append(data)
					break
				elif element == 'negative':
					message = message.strip()
					data = [message, -1]
					results.append(data)
					break
				else:
					message = message + " " + element
					message = message.replace("\n", " ")
	return results


train_data = unpack_data(train_file)
test_data  = unpack_data(test_file)
print(f"There are {len(train_data)} elements in the training set.")
print(f"There are {len(test_data)} elements in the testing set.")

sentences = []

for datum in train_data:
	message, sentiment = datum
	sentences.append(message)
for datum in test_data:
	message, sentiment = datum
	sentences.append(message)

tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                              tags=[str(i)]) for i, doc in enumerate(sentences)]

model = Doc2Vec(vector_size=150, min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

x_training_vectors = []
y_training_vectors = []
msg_training_vectors = []

x_testing_vectors = []
y_testing_vectors = []
msg_testing_vectors = []

for i, datum in enumerate(train_data):
	message, sentiment = datum
	vector = model.infer_vector(word_tokenize(message.lower()))
	#print(f"Message: {message} :: Sentiment: {sentiment} :: Vector : {vector}")
	x_training_vectors.append(vector)
	msg_training_vectors.append(message)
	if sentiment == -1:
		y_training_vectors.append([1, 0, 0])
	elif sentiment == 0:
		y_training_vectors.append([0, 1, 0])
	elif sentiment == 1:
		y_training_vectors.append([0, 0, 1])

for i, datum in enumerate(test_data):
	message, sentiment = datum
	vector = model.infer_vector(word_tokenize(message.lower()))
	#print(f"Message: {message} :: Sentiment: {sentiment} :: Vector : {vector}")
	x_testing_vectors.append(vector)
	msg_testing_vectors.append(message)
	if sentiment == -1:
		y_testing_vectors.append([1, 0, 0])
	elif sentiment == 0:
		y_testing_vectors.append([0, 1, 0])
	elif sentiment == 1:
		y_testing_vectors.append([0, 0, 1])

x_training_vectors = np.array(x_training_vectors)
y_training_vectors = np.array(y_training_vectors)

x_testing_vectors = np.array(x_testing_vectors)
y_testing_vectors = np.array(y_testing_vectors)

def conduct_training(model_depth, model_width, dropoff=False, narrow=True, annealing=False, epochs=16):

	model = tf.keras.Sequential(name='classification')

	#model.add(tf.keras.layers.Dense(model_width, name='dense_0', input_shape=(150,)))
	#model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid, name='activation_0'))

	model.add(tf.keras.layers.Normalization(name='normalization_0', input_shape=(150,)))
	
	layer_width = model_width
	for i in range(model_depth):
		model.add(tf.keras.layers.Dense(layer_width, name=f'dense_{i}'))
		if (dropoff):
			model.add(tf.keras.layers.Dropout(0.25))
		model.add(tf.keras.layers.Activation(tf.keras.activations.relu, name=f'activation_{i}'))
		if (narrow):
			layer_width = int(layer_width/2)

	model.add(tf.keras.layers.Dense(3, name=f'dense_{i+1}'))
	model.add(tf.keras.layers.Activation(tf.keras.activations.softmax, name=f'activation_{i+2}'))

	model.summary()

	if annealing: #https://keras.io/api/optimizers/ 
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    	initial_learning_rate=1e-2,
    	decay_steps=10000,
    	decay_rate=0.9)
		optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
	else:
		model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])

	model.fit(x_training_vectors, y_training_vectors, epochs=epochs, batch_size=64)

	test_loss, test_accuracy = model.evaluate(x_testing_vectors, y_testing_vectors)
	print(f'Test accuracy: {test_accuracy} with {model_depth} layers each being {model_width} nodes.')
	return test_accuracy

layer_depths = [5, 7]  
layer_widths = [256*2, 256*4, 256*8 ]
epochs = [16, 24, 32]

depth, width = 5, 512
f = open("training_results_new.txt", "a")
f.write(f"depth,width,epochs,annealing,accuracy\n")
f.close()
for i in range(1):
	for epoch in epochs:
		accuracy = conduct_training(depth, width, epochs=epoch, annealing=False)
		f = open("training_results.txt", "a")
		f.write(f"{depth},{width},{epoch},False,{accuracy}\n")
		f.close()

		accuracy = conduct_training(depth, width, epochs=epoch, annealing=True)
		f = open("training_results.txt", "a")
		f.write(f"{depth},{width},{epoch},True,{accuracy}\n")
		f.close()