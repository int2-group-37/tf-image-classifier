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

print(len(training_set))

# normalization_layer = keras.layers.Rescaling(1./255)

# normalized_dataset = training_set.map(lambda x, y: (normalization_layer(x),y))
# images, labels = next(iter(normalized_dataset))

IMAGE_RES = 224


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


BATCH_SIZE = 64

NUM_TRAINING_DATA = 1020

train_batches = training_set.cache().shuffle(
    NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(
    format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

model = tf.keras.models.load_model('Saved_Model/Current_Model')

# The following evaluates the model on the test data
model.evaluate(test_batches)
