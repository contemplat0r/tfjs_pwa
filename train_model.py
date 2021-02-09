#!/usr/bin/env python
# coding: utf-8

import pathlib
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflowjs as tfjs


#tfjs_target_dir = 'tfjs_target_dir'
tfjs_models_dir = 'tf_models'


data_root = pathlib.Path('../data_tf_dataset')

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))


all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)


label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

all_images_as_np = np.array([tf.image.resize(mpimg.imread(image_path), [256, 256]) for image_path in all_image_paths])


all_image_labels_as_np = np.array(all_image_labels)


all_index = list(range(all_image_labels_as_np.shape[0]))

random.shuffle(all_index)

all_index_random_train = all_index[0:(3 * len(all_index) // 4)]
all_index_random_test = all_index[(3 * len(all_index) // 4):]


print(len(all_index_random_train), len(all_index_random_test))

all_images_as_np_shuffled = all_images_as_np[all_index]


all_labels_as_np_shuffled = all_image_labels_as_np[all_index]


x_train = all_images_as_np_shuffled[0:(3 * len(all_index) // 4)]
x_test = all_images_as_np_shuffled[(3 * len(all_index) // 4):]


y_train = all_labels_as_np_shuffled[0:(3 * len(all_index) // 4)]
y_test = all_labels_as_np_shuffled[(3 * len(all_index) // 4):]

num_classes = 2
input_shape = (256, 256, 3)


x_train = x_train / 255
x_test = x_test / 255

x_train_exp = np.expand_dims(x_train, -1)
x_test_exp = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape, y_test.shape)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 16
epochs = 6

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25)

tfjs.converters.save_keras_model(model, tfjs_models_dir)