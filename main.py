import tensorflow as tf
import tensorflow_datasets as tfds
import os


dataset, dataset_info = tfds.load('oxford_flowers102', data_dir=(os.getcwd() + '/dataset'), with_info=True, as_supervised=True)
test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

print(len(training_set))