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

train_batches = training_set.cache().shuffle(NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

num_classes = 102

tf.random.set_seed(22)

model = keras.models.Sequential([

  keras.layers.RandomFlip('horizontal'),
  keras.layers.RandomRotation(factor = 0.2, fill_mode = 'nearest'),
  keras.layers.RandomZoom(0.5),
  keras.layers.RandomContrast(0.7),

  keras.layers.GaussianNoise(stddev=0.001),

  keras.layers.Conv2D(filters = 32, kernel_size = (3, 3),strides=(1,1), activation = 'relu'),
  keras.layers.MaxPool2D(),

  keras.layers.Conv2D(filters = 32, kernel_size = (3, 3),strides=(1,1), activation = 'relu'),
  keras.layers.MaxPool2D(),

  keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),strides=(1,1), activation = 'relu'),
  keras.layers.MaxPool2D(pool_size=(3,3)),

  keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),strides=(1,1), activation = 'relu'),
  keras.layers.MaxPool2D(pool_size=(3,3)),

  keras.layers.Flatten(),
  keras.layers.Dropout(0.45),
  keras.layers.Dense(256, activation = 'relu'),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(102, activation = 'softmax')
])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95,
                              patience=20, min_lr=0.00001)

mcp_save = keras.callbacks.ModelCheckpoint('Saved_Model/Checkpoint_Model', save_best_only=True, monitor='val_accuracy', mode='max')

model.compile(
  optimizer='adam',
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

history = model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=1000,
    callbacks=[reduce_lr,mcp_save]
    )

model.save('Saved_Model/Final_Model')
model.evaluate(test_batches)