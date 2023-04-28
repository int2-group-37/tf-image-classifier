import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import os
import numpy as np


dataset, dataset_info = tfds.load('oxford_flowers102', data_dir=(os.getcwd() + '/dataset'), with_info=True, as_supervised=True)
test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

print(len(training_set))

#normalization_layer = keras.layers.Rescaling(1./255)

#normalized_dataset = training_set.map(lambda x, y: (normalization_layer(x),y))
#images, labels = next(iter(normalized_dataset))

IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

NUM_TRAINING_DATA = 1020

train_batches = training_set.cache().shuffle(NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)


#autotune = tf.data.AUTOTUNE
#Snormalized_dataset = normalized_dataset.cache().prefetch(buffer_size=autotune)

num_classes = 102

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_batches,
  validation_data=validation_batches,
  epochs=3
)