#!/usr/bin/env python
# coding: utf-8

# # 카글 텍스트 분류 - 합성곱 신경망 활용 접근방법




import sys
import os
import numpy as np
import json

from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence





DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TEST_INPUT_DATA = 'test_input.npy'
TEST_ID_DATA = 'test_id.npy'

DATA_CONFIGS = 'data_configs.json'

train_input_data = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
train_label_data = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))
test_input_data = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb'))

with open(DATA_IN_PATH + DATA_CONFIGS, 'r') as f:
    prepro_configs = json.load(f)
    print(prepro_configs.keys())





# 파라메터 변수
RNG_SEED = 1234
BATCH_SIZE = 16
NUM_EPOCHS = 3
VOCAB_SIZE = prepro_configs['vocab_size'] + 1
EMB_SIZE = 128
VALID_SPLIT = 0.2

train_input, eval_input, train_label, eval_label = train_test_split(train_input_data, train_label_data, test_size=VALID_SPLIT, random_state=RNG_SEED)


# ## tf.data 세팅




def mapping_fn(X, Y=None):
    input, label = {'x': X}, Y
    return input, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label))
    dataset = dataset.shuffle(buffer_size=len(train_input))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(count=NUM_EPOCHS)

    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((eval_input, eval_label))
    dataset = dataset.shuffle(buffer_size=len(eval_input))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(mapping_fn)

    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()


# ## 모델 세팅




# 모델에 대한 메인 부분입니다.


def model_fn(features, labels, mode):

    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    #embedding layer를 선언합니다.
    embedding_layer = keras.layers.Embedding(
                    VOCAB_SIZE,
                    EMB_SIZE)(features['x'])
    
    # embedding layer에 대한 output에 대해 dropout을 취합니다.
    dropout_emb = keras.layers.Dropout(rate=0.5)(embedding_layer)

    ## filters = 128이고 kernel_size = 3,4,5입니다.
    ## 길이가 3,4,5인 128개의 다른 필터를 생성합니다. 3,4,5 gram의 효과처럼 다양한 각도에서 문장을 보는 효과가 있습니다.
    ## conv1d는 (배치사이즈, 길이, 채널)로 입력값을 받는데, 배치사이즈: 문장 숫자 | 길이: 각 문장의 단어의 개수 | 채널: 임베딩 출력 차원수임
    
    conv1 = keras.layers.Conv1D(
         filters=128,
         kernel_size=3,
        padding='valid',
         activation=tf.nn.relu)(dropout_emb)
    
    pool1 = keras.layers.GlobalMaxPool1D()(conv1)

    conv2 = keras.layers.Conv1D(
         filters=128,
         kernel_size=4,
        padding='valid',
         activation=tf.nn.relu)(dropout_emb)
    
    pool2 = keras.layers.GlobalMaxPool1D()(conv2)
    
    conv3 = keras.layers.Conv1D(
         filters=128,
         kernel_size=5,
        padding='valid',
         activation=tf.nn.relu)(dropout_emb)
    pool3 = keras.layers.GlobalMaxPool1D()(conv3)
    
    concat = keras.layers.concatenate([pool1, pool2, pool3]) #3,4,5gram이후 모아주기
    
    hidden = keras.layers.Dense(250, activation=tf.nn.relu)(concat)
    dropout_hidden = keras.layers.Dropout(rate=0.5)(hidden)
    logits = keras.layers.Dense(1, name='logits')(dropout_hidden)
    logits = tf.squeeze(logits, axis=-1)
    
    #최종적으로 학습, 평가, 테스트의 단계로 나누어 활용
    
    if PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'prob': tf.nn.sigmoid(logits)
            }
        )
        
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    if EVAL:
        pred = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})
        
    if TRAIN:
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss = loss)





model_dir = os.path.join(os.getcwd(), "data_out/checkpoint/cnn/")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig(save_checkpoints_steps=200, keep_checkpoint_max=2,
                                    log_step_count_steps=400)

 #에스티메이터 객체 생성
cnn_est = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config_tf)
cnn_est.train(train_input_fn) #학습하기
cnn_est.evaluate(eval_input_fn) #평가하기





test_input_data = np.load(open(DATA_IN_PATH + TEST_INPUT_DATA, 'rb')) 
ids = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'))

predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":test_input_data}, shuffle=False)

predictions = np.array([p['prob'] for p in cnn_est.predict(input_fn=predict_input_fn)])





output = pd.DataFrame( data={"id": ids, "sentiment": predictions} )

output.to_csv( DATA_OUT_PATH + "Bag_of_Words_model_test.csv", index=False, quoting=3 )

