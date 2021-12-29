'''
CMEMS다운로드
MOTU를 이용해 다운받음
초안: 2021.12.18
'''


def download_CMEMS(path_base_dir, code, date_str):
    '''
    https://help.marine.copernicus.eu/en/articles/4899195-how-to-write-and-run-a-script-to-download-a-subset-of-a-dataset-from-the-copernicus-marine-data-store
    python -m pip install motuclient==1.8.4 --no-cache-dir
    '''
    import subprocess
    import os
    import datetime as dt

    ### 날짜 정리
    yyyy = date_str[:4]
    mm = date_str[5:7]
    dd = date_str[8:10]
    HH = date_str[11:13]
    MM = date_str[14:16]
    SS = date_str[17:19]
    date_filename=yyyy+mm+dd+'_'+HH+MM+SS

    ### 접속계정
    user_name='Hjeon'
    password='#Gunia4ever'

    ### 서비스 목록
    service_ids={'PHY_NRT_OBS':'SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046',\
                 'PHY_REP_OBS':'SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047',\
                 'PHY_FCST':'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS',\
                 'BIO_FCST':'GLOBAL_ANALYSIS_FORECAST_BIO_001_028-TDS', \
                 'WAV_FCST': 'GLOBAL_ANALYSIS_FORECAST_WAV_001_027-TDS', \
                 'WIND_NRT_OBS':'WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004-TDS'}
    product_ids={'PHY_NRT_OBS':'dataset-duacs-nrt-global-merged-allsat-phy-l4',\
                 'PHY_REP_OBS':'dataset-duacs-rep-global-merged-allsat-phy-l4',\
                 'PHY_FCST':'global-analysis-forecast-phy-001-024',\
                 'BIO_FCST':'global-analysis-forecast-bio-001-028', \
                 'WAV_FCST': 'global-analysis-forecast-wav-001-027', \
                 'WIND_NRT_OBS':'CERSAT-GLO-BLENDED_WIND_L4-V6-OBS_FULL_TIME_SERIE'}
    file_sizes={'PHY_FCST':160e6}

    ### 저장경로 지정
    path_out_dir = path_base_dir+'/'+service_ids[code]+'/'+product_ids[code]+'/'+yyyy+'/'+mm
    path_final = path_out_dir+'/'+product_ids[code] + '_'+ date_filename +'.nc'

    cmd = 'python -m motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id ' + service_ids[
        code] + \
          ' --product-id ' + product_ids[code] + \
          ' --date-min "' + date_str + '" --date-max "' + date_str + \
          '" --depth-min ' + str(0.493) + ' --depth-max ' + str(0.4942) + \
          ' --out-dir "' + path_out_dir + \
          '" --out-name "' + product_ids[code] + '_' + date_filename + '.nc"' + \
          ' --user ' + user_name + ' --pwd ' + password + ''

    ### 파일 다운로드여부 및 용량 확인 후, 다운로드 개시
    if os.path.isfile(path_final)==True: # 동일한 파일명이 있을지라도
        if os.path.getsize(path_final) < file_sizes[code]: # 용량이 이보다 작다면 다운로드
            os.makedirs(path_out_dir, exist_ok=True)
            subprocess.run(cmd)
    else: # 동일한 파일명이 없다면 다운로드
        os.makedirs(path_out_dir, exist_ok=True)
        subprocess.run(cmd)

import datetime as dt
import numpy as np
path_base_dir='Z:/전호군/Satellite/CMEMS'

### 현재 UTC시각은?
hour=(dt.datetime.today()-dt.timedelta(hours=9)).hour

### PHY_FCST
if hour == 12:
    download_CMEMS(path_base_dir, code='PHY_FCST', date_str='2021-12-16 12:00:00') # 오늘날짜+9일 까지 # 24시간 간격
### BIO_FCST
# download_CMEMS(path_base_dir, code='BIO_FCST', date_str='2021-11-15 12:00:00') # 오늘날짜+6일 까지 # 서비스 안됨 # 24시간 간격
### WAV_FCST
if hour in np.arange(0,24,3):
    download_CMEMS(path_base_dir, code='WAV_FCST', date_str='2021-11-15 00:00:00') # 오늘날짜+9일 까지 # 3시간 간격 0,3,6...,21
### WIND_NRT
if hour in np.arange(0,24,6):
    download_CMEMS(path_base_dir, code='WIND_NRT_OBS', date_str='2021-11-15 06:00:00') # 오늘날짜-45일 까지 # 6시간 간격 0,6,12,18

