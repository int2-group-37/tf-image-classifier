from platform import platform
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

dataset, dataset_info = tfds.load('oxford_flowers102', data_dir=(
    os.getcwd() + '/dataset'), with_info=True, as_supervised=True)
test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

NUM_TRAINING_DATA = 1020

train_batches = training_set.cache().shuffle(NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

num_classes = 102

model = tf.keras.models.Sequential([

  #tf.keras.layers.RandomFlip('vertical', input_shape = (224, 224, 3)),
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(factor = 0.2, fill_mode = 'nearest'),
  tf.keras.layers.RandomZoom(0.3),
  tf.keras.layers.RandomContrast(0.3),

  tf.keras.layers.GaussianNoise(stddev=0.001),
  tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3),strides=(1,1), activation = 'relu'),
  tf.keras.layers.MaxPool2D(),

  tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),strides=(1,1), activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size=(3,3)),

  tf.keras.layers.Flatten(),
  #tf.keras.layers.Dense(1024, activation = 'relu'),
  tf.keras.layers.Dropout(0.45),
  tf.keras.layers.Dense(256, activation = 'relu'),
  tf.keras.layers.Dropout(0.45),
  tf.keras.layers.Dense(102, activation = 'softmax')
])

# UPDATED: Factor: 0.8 -> 0.2, patience: 10 -> 20
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95,
                              patience=20, min_lr=0.00001)

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

history = model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=200,
    callbacks=[reduce_lr]
    )

model.evaluate(test_batches)