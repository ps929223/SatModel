import h5py
import numpy as np
from spectral.io import envi
import matplotlib.pyplot as plt

path_input_Mask_hdr = 'E:/02_Satellite/GOCI1/GOCI1_Landmask/GDPS_MASK.hdr'
path_input_Mask_img = 'E:/02_Satellite/GOCI1/GOCI1_Landmask/GDPS_MASK.in'
path_input_Chl='E:/02_Satellite/GOCI1/CHL/2018/03/COMS_GOCI_L2A_GA_20180316051640.CHL.he5'

### Mask 읽는 파트
img = envi.open(path_input_Mask_hdr, path_input_Mask_img) # envi 함수로 파일을 읽음
img = img[:,:,:] # 3차원 BSQ형태로 읽어진 것을 2차원 Array로 변환
r,c,d=img.shape
img = img.reshape(r,c)
# 해양위성센터에서 제공받은 Mask파일은 정수 0,2,130으로 구성되어 있으며 각각 2: Land, 0: Sea, 130: Coastline을 나타냄
img[img==130]=1 # 130은 값이 0,2와 너무 차이나므로 1을 대신 할당함

plt.figure(1)
plt.imshow(img, cmap='jet') # colormap을 jet로 설정하고 이미지를 띄움
plt.colorbar()
plt.grid() # 격자 그리기


### Chl 읽는 파트
# import matplotlib.pyplot as plt
data = h5py.File(path_input_Chl, mode='r')
data['HDFEOS/POINTS'].keys()
CHL =np.array(data['HDFEOS/GRIDS/Image Data/Data Fields/CHL Image Pixel Values'])

plt.figure(2, figsize=(8,6))
# CHL 자료 그대로 출력
plt.subplot(2,2,1)
plt.imshow(CHL)
plt.colorbar()

# CHL값 0이하인 경우 NaN할당하고 출력
# NaN에는 결측치가 포함됨
plt.subplot(2,2,2)
CHL[CHL<=0]=np.nan
plt.imshow(CHL)
plt.colorbar()

## 이상값을 고려하여 상위 99% 초과는 삭제하여 출력
plt.subplot(2,2,3)
min,max=np.nanquantile(CHL, [0, 0.99])
plt.imshow(CHL, vmin=min, vmax=max)
plt.colorbar()

## 여백 없애기
plt.tight_layout()

## 해상영역 외는 -1 할당하고 출력
plt.subplot(2,2,4)
CHL[img!=0]=-1
plt.imshow(CHL, vmin=-1, vmax=max)
plt.colorbar()









