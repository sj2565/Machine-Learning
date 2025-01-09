import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split

DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
INPUT_TRAIN_DATA = 'nsmc_train_input.npy'
LABEL_TRAIN_DATA = 'nsmc_train_label.npy'
DATA_CONFIGS = 'data_configs.json'

input_data = np.load(open(DATA_IN_PATH + INPUT_TRAIN_DATA, 'rb'))
label_data = np.load(open(DATA_IN_PATH + LABEL_TRAIN_DATA, 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))

TEST_SPLIT = 0.1
RNG_SEED = 13371447
VOCAB_SIZE = prepro_configs['vocab_size'] + 1
EMB_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 1

input_train, input_eval, label_train, label_eval = train_test_split(input_data, label_data, test_size=TEST_SPLIT,
                                                                    random_state=RNG_SEED)


def mapping_fn(X, Y):
    input, label = {'x': X}, Y
    return input, label


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_train, label_train))
    dataset = dataset.shuffle(buffer_size=len(input_train))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((input_eval, label_eval))
    dataset = dataset.shuffle(buffer_size=len(input_eval))
    dataset = dataset.batch(16)
    dataset = dataset.map(mapping_fn)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def model_fn(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE,
        EMB_SIZE)(features['x'])

    dropout_emb = tf.keras.layers.Dropout(rate=0.2)(embedding_layer)

    conv = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu)(dropout_emb)

    pool = tf.keras.layers.GlobalMaxPool1D()(conv)

    hidden = tf.keras.layers.Dense(units=250, activation=tf.nn.relu)(pool)

    dropout_hidden = tf.keras.layers.Dropout(rate=0.2)(hidden, training=TRAIN)
    logits = tf.keras.layers.Dense(units=1)(dropout_hidden)

    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    if TRAIN:
        global_step = tf.train.get_global_step()
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)

    elif EVAL:
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

    elif PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.nn.sigmoid(logits),
            }
        )


est = tf.estimator.Estimator(model_fn, model_dir="data_out/checkpoint/cnn_model")

time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")

est.train(train_input_fn)
#
# time_end = datetime.utcnow()
# print(".......................................")
# print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
# print("")
# time_elapsed = time_end - time_start
# print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
#
# valid = est.evaluate(eval_input_fn)
#
# INPUT_TEST_DATA = 'nsmc_test_input.npy'
# LABEL_TEST_DATA = 'nsmc_test_label.npy'
#
# test_input_data = np.load(open(DATA_IN_PATH + INPUT_TEST_DATA, 'rb'))
# test_label_data = np.load(open(DATA_IN_PATH + LABEL_TEST_DATA, 'rb'))
#
#
# def test_input_fn():
#     dataset = tf.data.Dataset.from_tensor_slices((test_input_data, test_label_data))
#     dataset = dataset.batch(16)
#     dataset = dataset.map(mapping_fn)
#     iterator = dataset.make_one_shot_iterator()
#
#     return iterator.get_next()
#
#
# predict = est.evaluate(test_input_fn)
#
print('finished')