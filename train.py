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

def greyscale(image,label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

BATCH_SIZE = 64

NUM_TRAINING_DATA = 1020

train_batches = training_set.cache().shuffle(NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)


# autotune = tf.data.AUTOTUNE
# Snormalized_dataset = normalized_dataset.cache().prefetch(buffer_size=autotune)

num_classes = 102


model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),

    tf.keras.layers.Conv2D(16, 3, activation='relu', padding="same",input_shape = (224,224,3)),
    tf.keras.layers.MaxPooling2D((2,2),padding="same"),

    tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2,2),padding="same"),

    tf.keras.layers.Conv2D(64, 3, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2,2),padding="same"),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes,kernel_regularizer=keras.regularizers.l2(0.001))
])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.8,patience=10,min_lr=0.00001)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=150,
    callbacks=[reduce_lr]
)


# This gets the input data's size if need (GET RID OF BEFORE SENDING IN)
"""
config = model.get_config()
print(config["layers"][0]["config"]["batch_input_shape"])
"""

# These are for when we are actually ready to run the model
model.save('Saved_Model/Current_Model')
#new_model = tf.keras.models.load_model('Saved_Model/Current_Model')

# The following evaluates the model on the test data
#model.evaluate(test_batches)