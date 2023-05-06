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

IMAGE_RES = 150


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label
"""
def greyscale(image,label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

def random_change_brightness(image, label):
    image = tf.image.stateless_random_brightness(image, 0.3, seed=(1,2))
    return image, label

def random_change_contrast(image, label):
    seed = (1, 2)
    image = tf.image.stateless_random_brightness(image, 0.2, seed)
    return image, label

def random_change_hue(image, label):
    seed = (1, 2)
    image = tf.image.stateless_random_hue(image, 0.2, seed)
    return image, label


trainset2 = training_set.map(random_change_brightness)
trainset3 = training_set.map(random_change_contrast)
trainset4 = training_set.map(random_change_hue)
training_set = training_set.concatenate(trainset2).concatenate(trainset3).concatenate(trainset4)
"""

BATCH_SIZE = 128
print(training_set.cardinality().numpy())
NUM_TRAINING_DATA = training_set.cardinality().numpy()

train_batches = training_set.cache().shuffle(NUM_TRAINING_DATA//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

test_batches = test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)



# autotune = tf.data.AUTOTUNE
# Snormalized_dataset = normalized_dataset.cache().prefetch(buffer_size=autotune)

num_classes = 102
loss2 = 900000


model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomContrast(factor=(0.1,0.2),seed=(1,2)),
    tf.keras.layers.RandomZoom(height_factor=(0, -0.3),width_factor=(0,-0.3),fill_mode="wrap"),
    tf.keras.layers.RandomRotation(0.5),

    tf.keras.layers.Conv2D(64, 3, activation='relu', padding="same",input_shape = (150,150,3)),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2,2),padding="same"),
    tf.keras.layers.Dropout(0.05),

    tf.keras.layers.Conv2D(128, 3, activation='relu', padding="same"),
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2,2),padding="same"),
    tf.keras.layers.Dropout(0.1),


    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2,2),padding="same"),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_classes,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001))
])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.8,patience=10,min_lr=0.00001)

model.compile(
    optimizer='adam',
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
