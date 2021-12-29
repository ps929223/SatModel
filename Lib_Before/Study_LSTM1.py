import pandas as pd
import numpy as np
import warnings
import os
import FinanceDataReader as fdr

warnings.filterwarnings('ignore')


### 데이터 호출
# 주식 코드를 활용해 데이터 불러오기
# 삼성전자 주식코드: 005930
STOCK_CODE = '005930'
# fdr 라이브러리를 활용해 삼성전자 데이터를 불러오세요
DF = fdr.DataReader(STOCK_CODE)

def minmaxscaleDF(DF, col_names):
    '''
    정규화(Normalization):minmaxscale
    col_names= ['Open', 'High', 'Low', 'Close', 'Volume']
    '''
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(DF[col_names])
    DF=pd.DataFrame(scaled, columns=col_names)
    return DF

def splitDF(DF, col_feature, col_target, test_size=0.2, random_state=0, shuffle=False):
    '''
    데이터 쪼개기
    col_feature=['Open','High','Low','Volume']
    col_target=['Close']
    test_size=0.2
    random_state=0
    shuffle=False
    '''
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = \
        train_test_split(DF[col_feature], DF[col_target], test_size=test_size, random_state=random_state, shuffle=shuffle)
    return x_train, x_test, y_train, y_test

def windowed_dataset(series, win_size, batch_size, shuffle):
    '''
    TF의 Dataset 활용해 시퀀스데이터셋을 함수로 구현
    series=y_train
    '''
    import tensorflow as tf
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(win_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(win_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

def hyperparams(y_train, y_test, win_size=20, batch_size=32):
    # trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋
    # window_size batch_size 각 데이터셋에 적용
    train_data = windowed_dataset(y_train, win_size, batch_size, True)
    test_data = windowed_dataset(y_test, win_size, batch_size, False)

    # X: (batch_size, window_size, feature)
    # Y: (batch_size, feature)
    return train_data, test_data


def gen_model(train_data, test_data, win_size=20, batch_size=32):
    '''
    Sequantial Model 구현
    '''
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    model = Sequential([
        # 1차원 feature map 생성합니다. filters는 32로, kernel_size는 5로 지정해주세요.
        Conv1D(filters=32, kernel_size=5,
               padding="causal",
               activation="relu",
               input_shape=[win_size, 1]),
        # LSTM과 Dense 레이러를 사용해주세요. 활성함수는 각각 tanh와 relu로 지정합니다.
        LSTM(16, activation='relu'),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    '''
    모델의 Compile. Loss는 Huber 함수, optimizer는 Adam 
    '''
    # Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
    loss = Huber()
    optimizer = Adam(0.0005)
    model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

    # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
    earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    # val_loss 기준 체크포인터도 생성합니다.
    filename = os.path.join('tmp', 'ckeckpointer.ckpt')
    checkpoint = ModelCheckpoint(filename,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    # callbacks로 앞에서 구현한 earlystopping과 checkpoint를 지정해주세요.
    history = model.fit(train_data,
                        validation_data=(test_data),
                        epochs=50,
                        callbacks=[checkpoint, earlystopping])
    return model