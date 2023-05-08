import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import os
import numpy as np


dataset, dataset_info = tfds.load('oxford_flowers102', data_dir=(
    os.getcwd() + '/dataset'), with_info=True, as_supervised=True)
test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']


IMAGE_RES = 224


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


BATCH_SIZE = 32

NUM_TRAINING_DATA = 1020

train_batches = training_set.cache().shuffle(
    NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(
    format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

modelToRun = input("\n\nEnter model name to run (or enter nothing to exit) >  Saved_Model/")

while modelToRun != "":

    model = tf.keras.models.load_model('Saved_Model/' + modelToRun)

    # The following evaluates the model on the test data
    model.evaluate(test_batches)
    
    modelToRun = input("\n\nEnter model name to run (or enter nothing to exit) >  Saved_Model/")

