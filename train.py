#!/usr/bin/python

import sys
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import matplotlib.pylab as plt
import numpy
import time


from keras import backend as K
if K.backend()=='tensorflow':
  K.set_image_dim_ordering("th")

inputWidth = 32 * 32 * 3
layerWidth = [ 96, 96, 192, 192, 192, 192 ]
inputShape = (3, 32, 32)
denseLayerWidth = 256
numClasses = 10 # cifar-10
convolutionSize = 3
batchSize = 128
epochs = 1000
numBatches = 5

checkpointPath = "/scratch/users/aheirich/cifar/cifar-10_{epoch:02d}_{val_acc:.2f}.best.hdf5"




def createModel():
  model = Sequential()
  for i in range(len(layerWidth)):
    width = layerWidth[i]
    if i == 0:
      model.add(Conv2D(inputWidth, convolutionSize,
                       activation='relu',
                       input_shape=inputShape))
    else:
      model.add(Conv2D(inputWidth, convolutionSize,
                       activation='relu'))
  model.add(Flatten())
  model.add(Dense(denseLayerWidth, activation='relu'))
  model.add(Dense(numClasses, activation='relu'))
  model.compile(loss='categorical_crossentropy',
                optimizer='Adadelta',
                metrics=['accuracy'])
  return model


def createModel2():
  model = Sequential()
  model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Convolution2D(96, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(96, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Convolution2D(192, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(192, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model



def plot_model_history(model_history):
  fig, axs = plt.subplots(1,2,figsize=(15,5))
  # summarize history for accuracy
  axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
  axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
  axs[0].set_title('Model Accuracy')
  axs[0].set_ylabel('Accuracy')
  axs[0].set_xlabel('Epoch')
  axs[0].set_xticks(numpy.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
  axs[0].legend(['train', 'val'], loc='best')
  # summarize history for loss
  axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
  axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
  axs[1].set_title('Model Loss')
  axs[1].set_ylabel('Loss')
  axs[1].set_xlabel('Epoch')
  axs[1].set_xticks(numpy.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
  axs[1].legend(['train', 'val'], loc='best')
  plt.show()



def accuracy(test_x, test_y, model):
  result = model.predict(test_x)
  predicted_class = numpy.argmax(result, axis=1)
  true_class = numpy.argmax(test_y, axis=1)
  num_correct = numpy.sum(predicted_class == true_class)
  accuracy = float(num_correct)/result.shape[0]
  return (accuracy * 100)





from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(numpy.unique(train_labels))


train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)


model = createModel()

checkpoint = ModelCheckpoint(checkpointPath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')


start = time.time()
print 'starting time', start
model_info = model.fit(train_features, train_labels,
        batch_size=batchSize,
        epochs=epochs,
        verbose=1,
        callbacks=[ checkpoint ],
        validation_data=(test_features, test_labels))
end = time.time()
print 'ending time', end
plot_model_history(model_info)


