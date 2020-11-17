import gzip
import numpy


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
              'Invalid magic number %d in MNIST label file: %s' %
              (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


def read_emnist(emnist_dir):

    TRAIN_IMAGES = emnist_dir+'/emnist-balanced-train-images-idx3-ubyte.gz'
    TRAIN_LABELS = emnist_dir+'/emnist-balanced-train-labels-idx1-ubyte.gz'
    TEST_IMAGES = emnist_dir+'/emnist-balanced-test-images-idx3-ubyte.gz'
    TEST_LABELS = emnist_dir+'/emnist-balanced-test-labels-idx1-ubyte.gz'
    MAPPING = emnist_dir+'/emnist-balanced-mapping.txt'

    train_images = extract_images(TRAIN_IMAGES)
    train_labels = extract_labels(TRAIN_LABELS)
    test_images = extract_images(TEST_IMAGES)
    test_labels = extract_labels(TEST_LABELS)

    with open(MAPPING, "r") as f:
        mapping = f.readlines()
        mapping = {str(x.split()[0]): str(x.split()[1]) for x in mapping}

    # Convert to float32
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Normalize
    train_images /= 255
    test_images /= 255

    # Output format: (28, 28, 1)
    return ((train_images, train_labels), (test_images, test_labels), mapping)

import pickle
import os
from keras.models import save_model

def save(model, mapping, model_name):
    os.makedirs(os.path.dirname(BASE_PROJECT_PATH+'/models/'+model_name+"/"), exist_ok=True)
    model_yaml_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.yaml'
    model_h5_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.h5'
    mapping_model_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'_mapping.p'



    model_yaml = model.to_yaml()
    with open(model_yaml_path, "w") as yaml_file:
        yaml_file.write(model_yaml)

    save_model(model, model_h5_path)

    pickle.dump(mapping, open(mapping_model_path, 'wb'))

    return

# BASE PROJECT PATH
BASE_PROJECT_PATH = "Base project path"

# AWS CONFIGS
AWS_ACCESS_KEY = "Access key"
AWS_SECRET_KEY = "Secret key"
S3_BUCKET = "Bucket name"

import argparse
import numpy
import tensorflow.keras as keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.applications.resnet50 import ResNet50


K.set_image_dim_ordering('th')

f = open("cnnTest_result.txt", 'w')
def build_net(training_data, model_name='model', epochs=10):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data

    # reshape to be [samples][pixels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # create model
    '''
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))'''

    '''
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides = (1,1), padding = 'same', input_shape=(1, 28, 28), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (1,1)))


    model.add(Conv2D(128, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (1,1)))


    model.add(Conv2D(256, (3, 3), strides = (1,1), padding ='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (1,1)))


    model.add(Conv2D(256, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (1,1)))

    
    model.add(Conv2D(512, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (1,1)))

    model.add(Conv2D(512, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (1,1)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))'''

    #resnet_path = './input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides = (1,1), padding = 'same', input_shape=(1, 28, 28), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3),  strides = (1,1), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200, verbose=1)
    #print(model.summary())
    f.write(model.summary())

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    #print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    f.write("Baseline Error: %.2f%%" % (100-scores[1]*100))

    save(model, mapping, model_name)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to .mat file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train on')
    parser.add_argument('-m', '--model', type=str, help='Model name')
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    training_data = read_emnist("./gzip")
    model = build_net(training_data, args.model, epochs=args.epochs)