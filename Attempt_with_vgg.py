import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import os
import numpy as np


dataset, dataset_info = tfds.load('oxford_flowers102', data_dir=(
    os.getcwd() + '/dataset'), with_info=True, as_supervised=True)
test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

print(len(training_set))

IMAGE_RES = 224


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label

BATCH_SIZE = 128
print(training_set.cardinality().numpy())
NUM_TRAINING_DATA = training_set.cardinality().numpy()

train_batches = training_set.cache().shuffle(NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)


num_classes = 102

model = tf.keras.Sequential([
    # data augmentation layers
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomContrast(factor=(0.05,0.25),seed=(1,2)),
    tf.keras.layers.RandomZoom(height_factor=(0, -0.15),width_factor=(0,-0.15),fill_mode="wrap"),
    tf.keras.layers.RandomRotation(0.2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding="same",input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides= 2,padding='valid'),
    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Conv2D(96, (3,3), activation='relu', padding="same"),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides= 2,padding='valid'),
    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Conv2D(160, (3,3), activation='relu', padding="same"),
    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides= 2,padding='valid'),
    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu',kernel_initializer='random_normal',bias_initializer='zeros'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax',kernel_initializer='random_normal',bias_initializer='zeros')
])


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.8,patience=50,min_lr=0.00001)

model.compile(
    optimizer='sgd',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

mcp_save = keras.callbacks.ModelCheckpoint('Saved_Model/Current_Model_Best', save_best_only=True, monitor='val_loss', mode='min')
model.fit(
    train_batches,
    validation_data=validation_batches,
    epochs=200,
    callbacks=[reduce_lr]
    #ADD BACK SAVE FUNCTION
)

model.summary()


# This gets the input data's size if need (GET RID OF BEFORE SENDING IN)
"""
config = model.get_config()
print(config["layers"][0]["config"]["batch_input_shape"])
"""

# These are for when we are actually ready to run the model
#model.save('Saved_Model/Current_Model')
#new_model = tf.keras.models.load_model('Saved_Model/Current_Model')

# The following evaluates the model on the test data
#model.evaluate(test_batches)