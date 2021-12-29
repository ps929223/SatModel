'''
LSTM Multivariate
https://ahnjg.tistory.com/33
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def get_standard_train(features, TRAIN_SPLIT):
    '''
    표준화: 평균을 빼고 각 피처의 표준 편차로 나눔으로써 스케일링
    '''
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    # 표준화
    dataset = (dataset-data_mean)/data_std
    return data_mean, data_std, dataset



'''
-----------------
Single step model
------------------
모델은 제공된 일부 이력을 기반으로 미래의 단일 지점을 예측하는 방법을 학습
'''

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    '''
    주어진 step_size를 기반으로 과거 관측치를 샘플링한다
    ## Train
    target=dataset[2]
    start_index=0
    end_index=TRAIN_SPLIT
    history_size=past_history
    target_size=future_target
    step=STEP
    single_step=True
    '''


    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for ii in range(start_index, end_index):
        indices = range(ii-history_size, ii, step)
        data.append(dataset[indices])

        if single_step:
          labels.append(target[ii+target_size])
        else:
          labels.append(target[ii:ii+target_size])

    return np.array(data), np.array(labels)


def create_time_steps(length):
  '''
  length=20
  '''
  return list(range(-length, 0))


def show_plot(plot_data, delta, title):
  '''
  ## Sample
  plot_data=[x_train_uni[0], y_train_uni[0]]
  delta=0
  title='Sample Example'
  ## Baseline
  plot_data=[x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])]
  delta=0
  title= 'Baseline Prediction Example'
  '''
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for ii, x in enumerate(plot_data):
    if ii:  ## ii>=0 인 경우, 과거값을 출력함
      plt.plot(future, plot_data[ii], marker[ii], markersize=10, label=labels[ii])
    else:  ## ii==None 인 경우, 예측값을 출력함
      plt.plot(time_steps, plot_data[ii].flatten(), marker[ii], label=labels[ii])
  plt.legend()
  plt.xlim([time_steps[0], (future + 5) * 2])
  plt.xlabel('Time-Step')
  return plt


def gen_multivariate_data(dataset, TRAIN_SPLIT, past_history, future_target, STEP):
    '''
    네트워크에 지난 5개월(past_history)의 데이터, 즉 매월 관측되는 5개의 관측치가 표시됩니다
    샘플링(STEP)은 1개월마다 수행됩니다
    Single step model의 경우 데이터 포인트의 레이블은 30개월 뒤(future target)의 CO2입니다
    이를 위한 레이블을 만들기 위해 30관찰 후 CO2가 사용됩니다

    past_history = 5
    future_target = 2
    STEP = 1
    '''
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

    print ('Single window of past history : {}'.format(x_train_single[0].shape))
    # --> 120 : 5일 * 24시간
    # --> 3 : 3개의 특성
    return x_train_single, y_train_single, x_val_single, y_val_single

def gen_multivarite_dataset(BATCH_SIZE, BUFFER_SIZE, x_train_single, y_train_single, x_val_single, y_val_single):
    ### 데이터 처리
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    return train_data_single, val_data_single

def gen_single_step_model(x_train_single):
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    return single_step_model

def fit_single_step_model(single_step_model, train_data_single, val_data_single, epochs, steps_per_epoch, validation_steps):
    # epochs = 100
    # steps_per_epoch = 30
    # validation_steps = 1
    single_step_history = single_step_model.fit(train_data_single, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch,
                                                validation_data=val_data_single,
                                                validation_steps=validation_steps)
    return single_step_history, single_step_model


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

path_csv='E:/Dokdo_DB/CO2/co2_lstm_test.csv'
df = pd.read_csv(path_csv)
print(df.head())


'데이터의 처음 50개 행은 train dataset이고 나머지는 validation dataset'

TRAIN_SPLIT = 50
# 재현성을 보장하기 위해 시드 설정.
tf.random.set_seed(13)

'feature 설정'
features_considered = ['Anmyeondo', 'Gosan', 'Ulleungdo'] # DF에 있는 거 모두 사용

features = df[features_considered]
features.index = df['Month']
print(features.head())

'시간에 따른 각 특성을 살펴 보겠습니다'
features.plot(subplots=True)

'표준화'
data_mean, data_std, dataset=get_standard_train(features, TRAIN_SPLIT)

'LSTM에 맞도록 "다:1" 형태의 데이터 분할'
x_train_single, y_train_single, x_val_single, y_val_single=\
    gen_multivariate_data(dataset, TRAIN_SPLIT, past_history=24, future_target=1, STEP=1)

'LSTM에 맞도록 데이터 변형'
train_data_single, val_data_single = \
    gen_multivarite_dataset(BATCH_SIZE=256, BUFFER_SIZE=10000,
                            x_train_single=x_train_single, y_train_single=y_train_single,
                            x_val_single=x_val_single, y_val_single=y_val_single)

'모델 생성'
single_step_model=gen_single_step_model(x_train_single)


'샘플 예측을 확인합니다'
for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)
# --> 5 : 뭘 의미하지..?
# --> 1 : 1개의 예측


'모델의 fitting'
single_step_history, single_step_model\
    =fit_single_step_model(single_step_model, train_data_single, val_data_single,\
                           epochs=10, steps_per_epoch=1, validation_steps=1)

plot_train_history(single_step_history,
                   'Single Step Training and validation loss')


'''
Single step 예측
'''
plot = show_plot([x[0][:, 0].numpy(), y[0].numpy(),
                single_step_model.predict(x)[0]], 0,
               'Single Step Prediction')
plot.show()
