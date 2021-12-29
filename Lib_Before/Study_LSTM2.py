'''
LSTM Univariate and Multivariate
https://ahnjg.tistory.com/33
'''

from __future__ import absolute_import, division, print_function, unicode_literals
# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

''' Figure창에 대한 설정 '''
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


'''
날씨 데이터 세트
Max Planck Institute for Biogeochemistry에 의해 기록 된 [날씨 시계열 데이터 세트]를 사용합니다
데이터 세트에는 대기 온도, 대기압 및 습도와 같은 14 가지 기능이 있습니다. 
효율성을 위해 2009 년과 2016 년 사이에 수집 된 데이터 만 사용합니다
'''
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
print(df.head())

'''
데이터는 10분마다 기록됩니다, 한 시간 동안 6개, 하루에 144개의 관측치가 포함됩니다
특정 시간이 주어졌을때 앞으로 6시간 동안의 온도를 예측한다고 가정해봅니다

예측을 하기 위해 5일간의 데이터를 사용하도록 합니다
모형을 학습하기 위해 720(5*144) 개의 관측값이 포함된 window를 만듭니다
이러한 구성을 많이 만들수 있기 때문에 데이터 세트를 실험하기에 적합합니다
아래 함수는 모델이 훈련 할 때 위에서 설명한 time window를 반환합니다
history_size : 과거 information window의 크기 입니다 (몇 개의 과거 데이터를 학습할것인지)
target_size : 예측해야하는 레이블 입니다. 얼마나 멀리있는 예측을 배워야 하는가이다.
'''

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  '''
  ## Train ##
  dataset=uni_data
  start_index=0
  end_index=TRAIN_SPLIT
  history_size=univariate_past_history
  target_size=univariate_future_target

  ## Valiation ##
  dataset=uni_data
  start_index=TRAIN_SPLIT
  end_index=None
  history_size=univariate_past_history
  target_size=univariate_future_target
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


'''
데이터의 처음 300,000개 행은 train dataset이고 나머지는 validation dataset입니다
2100일 분량의 train data에 해당합니다
'''

TRAIN_SPLIT = 300000
# 재현성을 보장하기 위해 시드 설정.
tf.random.set_seed(13)


'''
##############################
## 1부. Univariate 시계열 예측
##############################
'''

'''
단일 특성(온도)만 사용하여 모델을 학습하고 향후 해당 값을 예측하는 데 사용합니다
데이터 세트에서 온도만 추출합니다
'''

uni_data = df['T (degC)']
uni_data.index = df['Date Time']
print(uni_data.head())
# 시간에 따른 데이터를 관찰합니다
uni_data.plot(subplots=True)
uni_data = uni_data.values

'''
신경망을 훈련하기 전 기능을 확장하는 것이 중요하다
표준화는 평균을 빼고 각 피처의 표준 편차로 나눔으로써 스케일링을 수행하는 일반적인 방법이다
tf.keras.utils.normalize값은 [0,1]범위로 재조정 하는 방법을 사용할 수도 있다
Note: 평균 및 표준 편차는 훈련 데이터만을 사용하여 계산해야합니다
'''

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
# 데이터를 표준화합시다.
uni_data = (uni_data-uni_train_mean)/uni_train_std


'''
Univariate 모델에 대한 데이터를 만듭니다
1부에서는 모델에 최근 20개의 온도 관측치가 제공되며 다음 단계에서 온도를 예측하는 방법을 배워야합니다
'''

univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

'''
Univariate_data 함수가 반환하는 것입니다
'''
print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])

'''
가시화
'''
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
        if ii: ## ii>=0 인 경우, 과거값을 출력함
            plt.plot(future, plot_data[ii], marker[ii], markersize=10, label=labels[ii])
        else: ## ii==None 인 경우, 예측값을 출력함
            plt.plot(time_steps, plot_data[ii].flatten(), marker[ii], label=labels[ii])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


'네트워크에서 제공되는 정보는 파란색으로 표시되며 붉은 X에서 값을 예측해야 합니다'
show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')



'''
#################
# Baseline
#################
'''

'''
모델 학습을 진행하기 전 기준을 설정하겠습니다
입력 지점이 주어지면 모든 기록을보고 다음 지점이 마지막 20개 관측치의 평균이 될 것으로 예측합니다
'''
def baseline(history):
  return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')



'''
############################
# Recurrent Neural network
############################
'''

'''
RNN(Recurrent Neural Network)은 시계열 데이터에 적합한 신경 네트워크 유형입니다
RNN은 시계열을 단계별로 처리하여 지금까지 본 정보를 요약하여 내부 상태를 유지합니다
LSTM이라고 불리는 RNN의 Layer을 사용합니다
tf.data 데이터 세트를 셔플, 배치 및 캐시하는 데 사용하겠습니다
'''

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


'LSTM에 데이터의 입력 형태가 필요하다는 것을 알 수 있습니다'

def gen_model():
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

simple_lstm_model=gen_model()

'모델의 출력을 확인하기 위해 샘플 예측을 만들어 봅니다'
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

'''
모델을 훈련시킵니다
데이터 세트의 크기 때문에 시간 절약을 위해 각 epoch는 200step만 실행됩니다
'''
EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)


'''
간단한 LSTM 모델 예측
'''
ii=1
for x, y in val_univariate.take(3):
    plt.subplot(3,1,ii)
    plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
    ii = ii + 1



'''
################################
# 2부 : Multivariate 시계열 예측
################################
'''

'''
원본 데이터 세트에는 14개의 특성이 있습니다. 간단하게하기 위해 14개 중 3개만 고려합니다.
사용되는 특성은 대기 온도, 대기압, 공기 밀도입니다
더 많은 특성을 사용하려면 해당 특성 이름을 목록에 추가하세요.
'''

features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']

features = df[features_considered]
features.index = df['Date Time']
print(features.head())

'시간에 따른 각 특성을 살펴 보겠습니다'
features.plot(subplots=True)

'첫 단계는 훈련 데이터의 평균 및 표준 편차를 사용하여 데이터 세트를 표준화하는 것입니다'
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# 표준화
dataset = (dataset-data_mean)/data_std

'''
-----------------
Single step model
------------------
'''

'''
모델은 제공된 일부 이력을 기반으로 미래의 단일 지점을 예측하는 방법을 학습합니다
아래의 함수는 주어진 step_size를 기반으로 과거 관측치를 샘플링한다
'''

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
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


'''
네트워크에 지난 5일 동안의 데이터, 즉 매 시간마다 샘플링되는 720(=6*24*5)개의 관측치가 표시됩니다
60분 내에 급격한 변화가 예상되지 않으므로 샘플링은 1시간마다 수행됩니다
120(=24*5)개의 관측치는 지난 5일의 이력을 나타냅니다
Single step model의 경우 데이터 포인트의 레이블은 12 시간 뒤의 온도입니다. (12시간 뒤의 온도 예측)
이를 위한 레이블을 만들기 위해 72(6*12)관찰 후 온도가 사용됩니다
'''
past_history = 720
future_target = 72
STEP = 6

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

### 데이터 처리
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

def gen_model():
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    return single_step_model

single_step_model=gen_model()



'샘플 예측을 확인합니다'
for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)
# --> 256 : 뭘 의미하지..?
# --> 1 : 1개의 예측


## Model Fitting
single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


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


plot_train_history(single_step_history,
                   'Single Step Training and validation loss')


'''
Single step 예측
모델이 학습되었으므로 몇 가지 샘플을 만듭니다
온도를 예측하는 것이 목표이기 때문에 플롯은 과거 온도만 표시합니다
'''
ii=1
for x, y in val_data_single.take(3):
    plt.subplot(3,1,ii)
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
    plot.show()
    ii=ii+1




'''
#####################
# Multi step model
#####################
'''

'''
모델은 과거 히스토리가 주어지면 미래의 값 범위를 예측하는 법을 배워야합니다
하나의 미래 포인트만 예측하는 Single step model과 달리 Multi step model은 미래의 시퀀스를 예측합니다
Multi step model의 훈련 데이터는 다시 한 시간마다 샘플링 된 지난 5일 동안의 기록으로 구성됩니다
12시간 동안의 온도를 예측하는 방법을 학습해야합니다
10분마다 관측이 수행되므로 결과는 72(1x6x12)개의 예측입니다
데이터 세트를 적절히 다시 준비해야합니다.
'''

future_target = 72
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('Target temperature to predict : {}'.format(y_train_multi[0].shape))
# Single window of past history : (120, 3)
# --> 120 : 5일 * 24시간
# --> 3 : 3개의 특성
# Target temperature to predict : (72,)
# --> 72개의 예측 데이터

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


'샘플 데이터 포인트 플로팅'
def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
        plt.legend(loc='upper left')
        plt.show()
    plt.legend()

for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))

'''
이전 작업보다 조금 더 복잡하기 때문에 모델을 두 개의 LSTM 계층으로 구성합니다
마지막으로 72개의 예측이 이루어지므로, Dense layer는 72개의 예측을 출력합니다
'''

def gen_model():
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(72))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    return multi_step_model

multi_step_model=gen_model()


'학습하기 전에 모델이 어떻게 예측하는지 봅시다'
for x, y in val_data_multi.take(1):
    print(multi_step_model.predict(x).shape)
# (256, 72)
# --> 256 : ?
# --> 72 : 72개의 예측 (6(한시간 여섯개) * 12시간)

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


'Multi step 예측'
ii=1
for x, y in val_data_multi.take(3):
    plt.subplot(3,1,ii)
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    ii=ii+1





