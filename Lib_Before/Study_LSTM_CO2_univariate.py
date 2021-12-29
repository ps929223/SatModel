'''
LSTM Univariate
https://ahnjg.tistory.com/33
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


'''
데이터 읽기
'''

path_csv='E:/Dokdo_DB/CO2/co2_lstm_test.csv'
df = pd.read_csv(path_csv)
print(df.head())


'''
울릉도 CO2만 학습, 향후 해당 값을 예측하는 데 사용
'''

uni_data = df['Ulleungdo']
Time = df['Month']
uni_data.plot(subplots=True)
uni_data = uni_data.values

'''
데이터의 처음 50개 행은 train dataset이고 나머지는 validation dataset
'''

TRAIN_SPLIT = 50
# 재현성을 보장하기 위해 시드 설정.
tf.random.set_seed(13)

'''
Window에서 과거데이터는 몇개? 몇번 뒤의 값을 예측할 것인지? 0:1번째
'''
past_history=5 # 과거데이터 개수
future_target=0 # 예측데이터 번째



def get_standard_train(uni_data, TRAIN_SPLIT):
  '''
  표준화: 평균을 빼고 각 피처의 표준 편차로 나눔으로써 스케일링
  '''
  uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
  uni_train_std = uni_data[:TRAIN_SPLIT].std()
  # 데이터를 표준화합시다.
  uni_data = (uni_data-uni_train_mean)/uni_train_std
  return uni_train_mean, uni_train_std, uni_data

uni_train_mean, uni_train_std, uni_data=get_standard_train(uni_data, TRAIN_SPLIT)



def univariate_data(dataset, start_index, end_index, history_size, target_size):
  '''

  '''
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None: # Validation 데이터의 경우
    end_index = len(dataset) - target_size

  for ii in range(start_index, end_index):
      # ii=21
    indices = range(ii-history_size, ii)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[ii+target_size])
  return np.array(data), np.array(labels)

def gen_univariate_data(uni_data, TRAIN_SPLIT, past_history, future_target):
  '''
  Univariate 모델에 대한 데이터를 생성
  최근 {past_history}개의 CO2 관측치가 제공되며 그 다음{future_target}단계에서 CO2를 예측
  past_history = 5
  future_target = 0
  '''
  x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                             past_history,
                                             future_target)
  x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                         past_history,
                                         future_target)

  print('Single window of past history')
  print(x_train_uni[0])
  print('Target temperature to predict')
  print(y_train_uni[0])

  return x_train_uni, y_train_uni, x_val_uni, y_val_uni


x_train_uni, y_train_uni, x_val_uni, y_val_uni=\
  gen_univariate_data(uni_data, TRAIN_SPLIT,past_history=past_history, future_target=future_target)

def gen_univarite_dataset(BATCH_SIZE, BUFFER_SIZE, x_train_uni, y_train_uni, x_val_uni, y_val_uni):
  '''
  BATCH_SIZE = 32
  BUFFER_SIZE = 5
  데이터를 shuffle, batch, cache 함(?)
  '''

  train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
  train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
  val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
  return train_univariate, val_univariate


train_univariate, val_univariate=\
  gen_univarite_dataset(BATCH_SIZE=32, BUFFER_SIZE=5,
                        x_train_uni=x_train_uni, y_train_uni=y_train_uni,
                        x_val_uni=x_val_uni, y_val_uni=y_val_uni)



def gen_simple_lstm_model(x_train_uni):
    ## 만약 아래 오류가 발생하면, Numpy 버젼을 바꾸어줌. numpy 1.20 이상버전일 때 발생하는 오류
    # Cannot convert a symbolic Tensor (lstm_1/strided_slice:0) to a numpy array.
    # pip install numpy==1.19.5

    simple_lstm_model = tf.keras.models.Sequential([
          # x_train_uni.shape = (299980, 20, 1)
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    simple_lstm_model.compile(optimizer='adam', loss='mae')
    return simple_lstm_model


simple_lstm_model=gen_simple_lstm_model(x_train_uni)



def fit_simple_lstm(simple_lstm_model, train_univariate, val_univariate, steps_per_epoch, epochs):
  '''
  모델을 훈련시킵니다
  steps_per_epoch = 1
  epochs = 10
  '''

  simple_lstm_model.fit(train_univariate, epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_univariate, validation_steps=50)
  return simple_lstm_model


simple_lstm_model= fit_simple_lstm(simple_lstm_model=simple_lstm_model, train_univariate=train_univariate,
                                   val_univariate=val_univariate, steps_per_epoch=1,epochs=100)



def create_time_steps(length,delta):
    return list(range(-length, delta))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0],delta)
  if delta:
    future = delta
  else:
    future = delta

  plt.title(title)
  for ii, x in enumerate(plot_data):
    if ii:  ## ii>0 인 경우, 예측값을 출력함
      plt.plot(future, plot_data[ii], marker[ii], markersize=10, label=labels[ii])
    else:  ## ii==0 인 경우, 과거값을 출력함
      plt.plot(time_steps[:len(time_steps)-delta], plot_data[ii].flatten(), marker[ii], label=labels[ii])
  plt.legend()
  plt.xlim([time_steps[0], (future + 5) * 2])
  plt.xlabel('Time-Step')
  return plt


'네트워크에서 제공되는 정보는 파란색으로 표시되며 붉은 X에서 값을 예측해야 합니다'
# show_plot(plot_data=[x_train_uni[0], y_train_uni[0]], delta=future_target, title='Sample Example')

def baseline(history):
  return np.mean(history)


'''
간단한 LSTM 모델 예측
'''

def recover(x,uni_train_mean, uni_train_std):
    '''
    표준화된 값을 원래값으로 복원
    '''
    return x*uni_train_std+uni_train_mean


ii=1
for x, y in val_univariate.take(3):
    plt.subplot(3,1,ii)
    setNo=ii-1
    plot = show_plot(plot_data=[recover(x[setNo].numpy(),uni_train_mean, uni_train_std),
                                recover(y[setNo].numpy(),uni_train_mean, uni_train_std),
                                recover(simple_lstm_model.predict(x)[setNo],uni_train_mean, uni_train_std)],
                     delta=future_target, title='Simple LSTM model')
    plt.show()
    plt.tight_layout()
    plt.grid()
    ii = ii + 1