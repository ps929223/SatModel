'''
CMEMS다운로드
MOTU를 이용해 다운받음

'''

import subprocess
import os

service_id={'PHY_NRT_OBS':'SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046',\
            'PHY_REP_OBS':'SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047'}
product_id={'PHY_NRT_OBS':'dataset-duacs-nrt-global-merged-allsat-phy-l4',\
            'PHY_REP_OBS':'dataset-duacs-rep-global-merged-allsat-phy-l4'}
variable='adt'
coord=[128, 138, 34, 43.7]
date_range=["2020-01-01 00:00:00","2021-01-01 00:00:00"]

user_name='Hjeon'
password='#Gunia4ever'

path_out_dir='Z:/전호군/Satellite/CMEMS/SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046/dataset-duacs-nrt-global-merged-allsat-phy-l4'
os.makedirs(path_out_dir, exist_ok=True)

cmd='python -m motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id ' + service_id['PHY_REP_OBS']+'-TDS' +\
    ' --product-id '+product_id['PHY_REP_OBS'] +\
    ' --longitude-min '+str(coord[0]) +' --longitude-max '+str(coord[1]) +\
    ' --latitude-min '+str(coord[2]) +' --latitude-max '+str(coord[3]) +\
    ' --date-min "'+date_range[0] + '" --date-max "'+date_range[1] +\
    '" --variable '+variable +\
    ' --out-dir "' + path_out_dir + \
    '" --out-name "' + service_id['PHY_REP_OBS'].split('_')[0] + \
                                '_'+variable + '_'+ date_range[0][:10] + '_' + date_range[1][:10]+'.nc' + \
    '" --user ' + user_name +' --pwd "' + password + '"'

subprocess.run(cmd)