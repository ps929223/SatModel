# -*- coding: utf-8 -*-
'''
FTP접속하여 CMEMS의 CHL을 다운받는 프로그램
2021.09.11
Auth: Hokun Jeon
Marine Bigdata Center
KIOST
'''

import os
from ftplib import FTP

'Path 설정'
path_from='/Core/OCEANCOLOUR_GLO_CHL_L4_NRT_OBSERVATIONS_009_033/dataset-oc-glo-bio-multi-l4-chl_interpolated_4km_daily-rt/'
path_to='D:/01_Model/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP'

def download_CHL(path_from,path_to):
	'Account '
	host = 'nrt.cmems-du.eu'  # ip
	id = 'hjeon'
	pw = '#Kordi1234'


	ftp = FTP(host)
	ftp.login(id,pw) #login info
	ftp.cwd(path_from) #directory

	# 가장 최신에 업로드된 디렉토리와 파일을 찾아서 다운로드 받기 위해
	# ex)월이 바뀌거나 연이 바뀌는 경우(데이터 Delay:  2일)

	## Year 목록 반환
	yy_dir_listing = []
	ftp.retrlines("LIST", yy_dir_listing.append)
	yy_words = yy_dir_listing[-1].split(None, 8)
	ftp.cwd(ftp.pwd() + '/' + yy_words[-1]) # 특정 Year내로 진입

	## Month 목록반환
	mm_dir_listing = []
	ftp.retrlines("LIST", mm_dir_listing.append)
	mm_words = mm_dir_listing[-1].split(None, 8)
	ftp.cwd(ftp.pwd() + '/' + mm_words[-1]) # 특정 Month내로 진입

	## file 목록반환
	file_listing = []
	ftp.retrlines("LIST", file_listing.append)
	words = file_listing[-1].split(None, 8)
	filename = words[-1].lstrip()

	path_Out = path_to + '/' + yy_words[-1] + '/' + mm_words[-1] #저장경로

	os.makedirs(path_Out, exist_ok=True)

	# download the file
	local_filename = os.path.join("%s" %path_Out, filename)
	lf = open(local_filename, "wb")
	ftp.retrbinary("RETR " + filename, lf.write, 8*1024)
	lf.close()
	ftp.quit()
