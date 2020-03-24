import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

from variables import*

def image_data_generator():
    train_datagen = ImageDataGenerator(
                                    rescale = rescale,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    horizontal_flip = True,
                                    validation_split=0.2
                                    )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    class_mode='binary',
                                    shuffle = True,
                                    subset='training')

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    class_mode='binary',
                                    shuffle = True,
                                    subset='validation')

    return train_generator, validation_generator

