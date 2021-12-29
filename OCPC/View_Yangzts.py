import pandas as pd
import matplotlib.pyplot as plt

path_file='D:/01_Model/KIO_OCPC/yangtze.dat'
data=pd.read_csv(path_file)

## 시간 Column 형태 변경
data['_']=' '
data['r']=':00:00'
data['datetime']=data['date'].astype(str) +  data['_'] +data['localtime'].astype(str) + data['r']
data['datetime']=pd.to_datetime(data['datetime'])

plt.plot(data.datetime, data.discharge/1e4)
plt.title('FW inflow from YTZ ('+ data.date[0] + '~' + data.date[data.index[-1]] +')')
plt.grid()
plt.xlabel('Local time')
plt.ylabel('Inflow '+r'$[10^4m^3/s]$')