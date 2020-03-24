import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import keras
from keras import backend as K
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Dropout

from variables import*
from util import image_data_generator

class CatVsDogs(object):
    def __init__(self):
        train_generator, validation_generator = image_data_generator()
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def classifier(self):
        model = Sequential()
        model.add(Conv2D(ofm,
                         kernel_size=kernal_size,
                         activation='relu',
                         input_shape=input_shape)
                         )
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(dense, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        self.model = model
        self.model.summary()

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= train_step,
                          validation_data=self.validation_generator,
                          validation_steps= val_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)
        print("Model Saved")

    def load_model(self):
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        print("Model Loaded")

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = loaded_model

    def predict(self):
        test_images, test_labels = next(self.validation_generator)
        Predictions = self.model.predict_generator(self.validation_generator,steps=val_step)
        P = Predictions > 0.5
        loss , accuracy = self.model.evaluate_generator(self.validation_generator, steps=val_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)
        print("Predictions : ",P)

if __name__ == "__main__":
    model = CatVsDogs()
    model.classifier()
    if os.path.exists(os.path.join(os.getcwd(),model_weights)):
        model.load_model()
    else:
        model.train()
        model.save_model()
    model.predict()
