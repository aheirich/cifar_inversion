#!/usr/bin/python

import sys
import keras
from keras.layers import Dense, Conv2D, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adadelta
import matplotlib.pylab as plt

inputWidth = 32 * 32 * 3
layerWidth = [ 96, 96, 192, 192, 192, 192 ]
inputShape = (32, 32, 3)
batchShape = (10000, 32, 32, 3)
outputShape = (10000, 10, 1, 1)
denseLayerWidth = 256
numClasses = 10 # cifar-10
convolutionSize = 3

def unpickle(file):
  import cPickle
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict

def createModel():
  model = Sequential()
  for i in range(len(layerWidth)):
    width = layerWidth[i]
    if i == 0:
      model.add(Conv2D(inputWidth, convolutionSize, activation='relu',
                       input_shape=inputShape))
    else:
      model.add(Conv2D(inputWidth, convolutionSize, activation='relu'))
  model.add(Flatten())
  model.add(Dense(denseLayerWidth, activation='relu'))
  model.add(Dense(numClasses, activation='relu'))
  model.compile(loss=keras.losses.mse,
                optimizer=Adadelta(),
                metrics=['accuracy'])
  return model

class AccuracyHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.acc = []
  def on_epoch_end(self, batch, logs={}):
    self.acc.append(logs.get('acc'))

def oneHot(labels):
  result = []
  for label in labels:
    y = [0] * numClasses
    y[label] = 1
    result.append(y)
  return result



numBatches = 5
training_data = []
for batch in range(numBatches):
  print 'get batch', batch
  filename = 'cifar-10-batches-py/data_batch_' + str(batch + 1)
  dict = unpickle(filename)
  x_train = dict['data'].reshape(batchShape)
  y_train = oneHot(dict['labels'])
  training_data.append((x_train, y_train))

test_data = unpickle('cifar-10-batches-py/test_batch')
x_test = test_data['data'].reshape(batchShape)
y_test = oneHot(test_data['labels'])

model = createModel()

batchSize = 128
epochs = 50
history = AccuracyHistory()

for batch in training_data:
  print 'training a batch'
  (x_train, y_train) = batch
  model.fit(x_train, y_train,
            batch_size=batchSize,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

