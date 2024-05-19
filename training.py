import itertools
import pickle

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

IMG_WIDTH = IMG_HEIGHT = 128
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 32
classes = [
    "Burger",
    "Donut",
    "Hot dog",
    "Pizza",
    "Sandwich"
]
def encode(x, y):
    y = tf.one_hot(y, NUM_CLASSES)
    return x, y


def perform_evaluation(model, ds, ds_name):
    print("Evaluation for " + ds_name)

    y_true = []
    y_pred = []

    for x, y in ds.batch(BATCH_SIZE):
        y_true.extend(y.numpy())
        predictions = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    ds = ds.map(encode)
    result = model.evaluate(ds.batch(BATCH_SIZE))
    print("Loss: " + str(result[0]))
    print("Accuracy: " + str(result[1]))
    print("F1-score: " + str(result[2]))
    print("False positives: " + str(result[3]))
    print("True negatives: " + str(result[4]))
    print("Specificity: " + str(result[4] / (result[4] + result[3])))

    plt.figure(figsize=(8, 8))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fmt = '.0f'
    # Add text annotations to the plot indicating the values in the cells
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="black")
    # Display the confusion matrix as an image with a colormap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.show()


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
test_ds = tf.data.TFRecordDataset(["./test.tfrecord"]).map(decode_fn)

train_ds_len = 0
for batch in train_ds:
    train_ds_len += 1
val_ds_len = 0
for batch in val_ds:
    val_ds_len += 1
test_ds_len = 0
for batch in test_ds:
    test_ds_len += 1
print("TRAIN_LEN=" + str(train_ds_len) + " VAL_LEN=" + str(val_ds_len))
print(train_ds_len // BATCH_SIZE)
print(val_ds_len // BATCH_SIZE)
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)),
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
    keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy', 'f1_score', 'false_positives', 'true_negatives'])
TRAINING = False
if TRAINING:
    # Dataset input
    # data = np.load('data_split3.npz')
    # train_x_set = data['train_x']
    # train_y_set = data['train_y']
    # val_x_set = data['val_x']
    # val_y_set = data['val_y']
    # test_x_set = data['test_x']
    # test_y_set = data['test_y']

    checkpoint_path = "saved_models/cp-{epoch:04d}.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    epochs = 30
    history = model.fit(
        train_ds.batch(BATCH_SIZE),
        validation_data=val_ds.batch(BATCH_SIZE),
        epochs=epochs,
        callbacks=[cp_callback],
        # steps_per_epoch=train_ds_len//BATCH_SIZE,
        # validation_steps=val_ds_len//BATCH_SIZE
    )

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, epochs + 1)

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

    print(val_acc)
    max_val_acc = max(val_acc)
    max_val_acc_epoch = val_acc.index(max_val_acc) + 1
    print("Max val_acc=" + str(max_val_acc) + ", achieved in epoch no." + str(max_val_acc_epoch))
else:
    model.load_weights("saved_models/cp-0029.weights.h5")
    perform_evaluation(model, train_ds, "TRAIN DATASET")
    perform_evaluation(model, val_ds, "VALIDATION DATASET")
    perform_evaluation(model, test_ds, "TEST DATASET")
