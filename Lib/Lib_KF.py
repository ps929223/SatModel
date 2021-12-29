import matplotlib.pyplot as plt
import numpy as np

def kalman_filter(z, Q, R, A, H):
    '''
    [도서] 실전시계열분석 p271-274
    칼만필터 R코드를 Python code 로 변환
    '''

    # Q=10; Q=np.eye(1)*Q

    dimState = np.shape(Q)[0] # Q의 row수
    # dimState = 1

    length_ts=np.shape(z)[0]
    # length_ts=100

    # 상태
    xhatminus = np.zeros(shape=(length_ts,dimState))
    xhat = np.zeros(shape=(length_ts,dimState))

    ## 공분산행렬
    Pminus = np.zeros(shape=(length_ts,dimState,dimState))
    P = np.zeros(shape=(length_ts,dimState,dimState))

    ## Kalman 이득
    K = np.zeros(shape=(length_ts,dimState))


    ## 초기 추측: 모두 0으로 시작
    xhat[1,:] = [0]*dimState
    P[1,:,:] = np.eye(dimState,dimState) # 단위행렬

    ## 시간 갱신
    for ii in range(1, length_ts):
        # ii =1
        ## 예측
        # A=np.array(1)
        # Q=np.array(10)
        xhatminus[ii,:] = np.dot(A,xhat[ii-1,:])
        Pminus[ii,:,:] = np.dot(np.dot(A,P[ii-1,:,:]),A.T)+Q

        ## 필터링
        K[ii,:] = np.dot(np.dot(Pminus[ii,:,:], H),
                         np.linalg.inv(np.dot(np.dot(H,Pminus[ii,:,:]),H.T)+R))
        xhat[ii,:] = xhatminus[ii,:]+np.dot(K[ii,:],
                                            (z[ii]-np.dot(H.T, xhatminus[ii,:])))
        P[ii,:,:] = np.dot(np.dot(np.eye(dimState,dimState)-K[ii,:],H.T),Pminus[ii,:,:])

    return {'xhat':xhat, 'xhatminus':xhatminus}


''' test code '''

## 노이즈 파라미터
R=10**2 # 측정분산. 알려진 것. 노이즈와 일관성 있게 설정
Q=10 # 과정의 분산. 최적화를 위해 조정되어야하는 Hyperparameter

## 동적 파라미터
A = np.array(1) # Xt = A * Xt-1 (사전 X가 나중 X에 얼마나 영향을 미치는지)
H = np.array(1) # Yt = H * Xt (상태를 측정으로 변환)

''' 데이터 생성 '''
## Time Stamp 갯수
length_ts=100

## 가속도
a = np.array([0.5]*length_ts) # 가속도

## 위치와 속도는 0부터 시작
x = np.array([0]*length_ts) # 위치
v = np.array([0]*length_ts) # 속도

## 위치, 속도 데이터생성
for ii in range(1, length_ts):
    x[ii] = v[ii-1] * 2 + x[ii-1] + 1/2 * a[ii-1] ** 2
    x[ii] = x[ii] + np.random.normal(scale=20,size=1) # scale: std
    v[ii] = v[ii-1] + 2 * a[ii-1]

## 관측값 생성
z = x + np.random.normal(scale=300,size=length_ts)

## 실행
Result = kalman_filter(z=z, Q=np.eye(1)*Q, R=R, A=A, H=H)

## plot
plt.plot(list(range(len(x))),x,c='tab:blue', linestyle='--', label='act')
plt.plot(list(range(len(x))),z,c='tab:orange', linestyle=':', label='obs')
plt.scatter(list(range(len(x))),Result['xhat'],marker='.', s=10, c='tab:green', label='filtered')
plt.scatter(list(range(len(x))),Result['xhatminus'],marker='.', s=10, c='tab:purple',label='pred')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Position')
plt.grid()


