'''
OPeNDAP 형태의 NANA 사이트로부터 SMAP SSS(염도)를 다운받는 코드
Auth: Hokun Jeon
Date: 2021.09.17
KIOST Marine Bigdata Center
'''

import requests

### 저장경로 설정 ###
path_output_dir='D:/01_Model/JPL/SMAP/SSS'

def download_SSS(yyyy,mm,path_output_dir):
    '''
    url='https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/smap/L3/JPL/V5.0/monthly/2021/SMAP_L3_SSS
    yyyy='2020'
    mm='09'
    '''
    url='https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/smap/L3/JPL/V5.0/monthly/'+yyyy\
        +'/SMAP_L3_SSS_'+yyyy+mm+'_MONTHLY_V5.0.nc'
    response = requests.get(url, stream=True)
    open(path_output_dir+'/'+url.split('/')[-1], 'wb').write(response.content)
    print('Downloaded :'+ url.split('/')[-1])


### 실행부 ###
# 다운받을 대상 월 설정
yyyy='2020'; mm='09'

# 다운로드 실행
download_SSS(yyyy,mm,path_output_dir)