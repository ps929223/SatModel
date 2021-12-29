'''
## Program 1
'독도종합정보시스템_업로드사진' 중에 워터마크가 있는  파일만 골라냄
: '독도사진전체'로 이송하고 독도사진전체의 이미지유사도 계산하기 위함
Hokun Jeon
Marine Bigdata Center
KIOST
2021.12.29
'''

import pandas as pd
import shutil, os

path_csv_dir='D:/00_연구/독도/04_DIS관리/20211228-2_영상및사진자료'
csv_name='watermark_images.csv'
path_img_dir='E:/Dokdo_DB/독도종합정보시스템_업로드사진/Original'

path_csv=path_csv_dir+'/'+csv_name
path_out_dir=path_csv_dir+'/watermark_img'
os.makedirs(path_out_dir, exist_ok=True)

DF=pd.read_csv(path_csv)

for ii in range(len(DF)):
    # ii=0
    shutil.copy(path_img_dir+'/'+DF['파일명'][ii],path_out_dir+'/'+DF['파일명'][ii])




'''
## Program 2
'독도종합정보시스템_업로드사진' 중에 워터마크가 있는 파일만 삭제하고,
대체 이미지를 해당 폴더에 다시 넣음
: 웹볼루션에 이미지 제공하기 위함
Hokun Jeon
Marine Bigdata Center
KIOST
2021.12.29
'''

import pandas as pd
import shutil, os

path_xls_dir='D:/00_연구/독도/04_DIS관리/20211228-2_영상및사진자료'
xls_name='독도종합정보시스템_영상 및 이미지 자료_20211229_호군.xlsx'
path_img_dir='E:/Dokdo_DB/독도종합정보시스템_업로드사진_v2_watermark제거/Original'
path_imgAll_dir='E:/Dokdo_DB/독도사진전체/Original'

path_xls=path_xls_dir+'/'+xls_name

DF=pd.read_excel(path_xls, sheet_name='이미지')

# 정중앙 KORDI 워터마크 파일 삭제
cond_rm=DF['마크&비고']=='정중앙 KORDI 워터마크'
list_rm=list(DF['파일명'][cond_rm])

for ii in range(len(list_rm)):
    if os.path.exists(path_img_dir+'/'+list_rm[ii]):
        os.remove(path_img_dir+'/'+list_rm[ii])

# 워터마크 있는 파일의 대체본을 복사해 넣음
cond_rep=DF['조치']=='대체'
list_rep=list(DF['대체이미지'][cond_rep])
path_out_dir='E:/Dokdo_DB/독도종합정보시스템_업로드사진_v2_watermark제거/Watermark대체이미지'
os.makedirs(path_out_dir, exist_ok=True)

for ii in range(len(list_rep)):
    if os.path.exists(path_imgAll_dir+'/'+list_rep[ii]):
        shutil.copy(path_imgAll_dir+'/'+list_rep[ii],path_out_dir+'/'+list_rep[ii])