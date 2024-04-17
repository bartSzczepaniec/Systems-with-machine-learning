import pickle

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
IMG_WIDTH = IMG_HEIGTH = 128
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 32


def decode_fn(record_bytes):
    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"x": tf.io.FixedLenFeature([128, 128, 3], dtype=tf.float32),
         "y": tf.io.FixedLenFeature([], dtype=tf.int64)}
    )
    return parsed_example["x"], parsed_example["y"]


train_ds = tf.data.TFRecordDataset(["./train.tfrecord"]).map(decode_fn)
val_ds = tf.data.TFRecordDataset(["./val.tfrecord"]).map(decode_fn)

train_ds_len = 0
for batch in train_ds:
    train_ds_len += 1
val_ds_len = 0
for batch in val_ds:
    val_ds_len += 1
print("TRAIN_LEN=" + str(train_ds_len) + " VAL_LEN=" + str(val_ds_len))
print(train_ds_len//BATCH_SIZE)
print(val_ds_len//BATCH_SIZE)
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# Dataset input
# data = np.load('data_split3.npz')
# train_x_set = data['train_x']
# train_y_set = data['train_y']
# val_x_set = data['val_x']
# val_y_set = data['val_y']
# test_x_set = data['test_x']
# test_y_set = data['test_y']

epochs = 30
history = model.fit(
    train_ds.batch(BATCH_SIZE),
    validation_data=val_ds.batch(BATCH_SIZE),
    epochs=epochs,
    #steps_per_epoch=train_ds_len//BATCH_SIZE,
    #validation_steps=val_ds_len//BATCH_SIZE
)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
