import keras
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Sequential
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
import time
import numpy as np
from PIL import Image
import json
from pathlib import Path
from build_dataset import load

img_rows = 32
img_cols = 32
channels = 3
num_classes = 2
input_shape = (img_rows, img_cols, channels)
epochs = 200
batch_size = 16
data_root = 'C:\\Mestrado\\Data\\'

def get_dataset():

    print('Loading Dataset\n')

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = load(data_root, img_rows, img_cols, channels)

    return x_train, y_train, x_test, y_test, x_valid, y_valid

def get_preprocessed_dataset():

    print('Preprocessing Dataset\n\n')
    
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_dataset()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_valid = x_valid.astype('float32')
    x_train /= 255
    x_test /= 255
    x_valid /= 255
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_valid.shape[0], 'valid samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    return x_train, y_train, x_test, y_test, x_valid, y_valid
    
def generate_optimizer():
    
    return keras.optimizers.Adam()

def compile_model(model):
    
    print('Compiling Dataset\n\n')
    model.compile(loss='categorical_crossentropy',
                  optimizer=generate_optimizer(),
                  metrics=['accuracy'])
                  
def generate_model():
    # check if model exists if exists then load model from saved state
    if Path('./models/convnet_model.json').is_file():
        print('Loading existing model\n\n')

        with open('./models/convnet_model.json') as file:
            model = keras.models.model_from_json(json.load(file))
            file.close()

        # likewise for model weight, if exists load from saved state
        if Path('./models/convnet_weights.h5').is_file():
            model.load_weights('./models/convnet_weights.h5')

        compile_model(model)

        return model

    print('Loading new model\n\n')

    model = Sequential()

    model.add(Conv2D(30, (5,5),padding='valid',input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' ))
    model.add(Dense(50, activation= 'relu' ))
    model.add(Dense(num_classes, activation= 'softmax' ))

    '''
    # Conv1 32 32 (3) => 30 30 (32)
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Conv2 30 30 (32) => 28 28 (32)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # Pool1 28 28 (32) => 14 14 (32)
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Conv3 14 14 (32) => 12 12 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # Conv4 12 12 (64) => 6 6 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # Pool2 6 6 (64) => 3 3 (64)
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FC layers 3 3 (64) => 576
    model.add(Flatten())
    # Dense1 576 => 256
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Dense2 256 => 10
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    '''

    compile_model(model)

    with open('./models/convnet_model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        outfile.close()

    return model
    
def train(model, x_train, y_train, x_test, y_test, x_valid, y_valid):
    
    print('Training model\n\n')

    epoch_count = 0
    while epoch_count < 5:
        epoch_count += 1
        print('Epoch count: ' + str(epoch_count) + '\n')
        model.fit(x_train, y_train, 
                    batch_size=batch_size,
                    epochs=epochs, 
                    validation_data=(x_valid, y_valid))
        print('Epoch {} done, saving model to file\n\n'.format(epoch_count))
        model.save_weights('./models/convnet_weights.h5')

    return model

def get_score(model, x_test, y_test, x_valid, y_valid):
    
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
def get_confusion_matrix(model, x_test, y_test, x_valid, y_valid):
    
    Y_pred = model.predict(x_test, verbose=2)
    y_pred = np.argmax(Y_pred, axis=1)

    for ix in range(num_classes):
        print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
    cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
    print(cm)

def main():
    
    print('GLAUCOMA CONVNET!\n\n')
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_preprocessed_dataset()
    model = generate_model()
    model = train(model, x_train, y_train, x_test, y_test, x_valid, y_valid)
    get_score(model, x_test, y_test, x_valid, y_valid)
    get_confusion_matrix(model, x_test, y_test, x_valid, y_valid)

if __name__ == "__main__":
    main()