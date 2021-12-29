import os, sys

# path_input_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP'
path_input_dir='E:/20_Product/DokdoVideoCapture'
path_output_dir= 'E:/20_Product/animation/DokdoVideoCapture'
os.makedirs(path_output_dir, exist_ok=True)
path_Lib='D:/programming/SatModel/Lib'
sys.path.append(path_Lib)

import Lib.lib_animation as ani

def month_GIF(path_month_dir, path_output_dir, fps):
    # path_month_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020/07'
    path_str=path_month_dir.split('/')
    output_file_name=path_str[-2]+path_str[-1]+'-'+path_str[-3]
    ani.ani_month_MP4(path_month_dir, path_output_dir, output_file_name+'.mp4', fps)

def year_GIF(path_year_dir, path_output_dir, fps):
    # path_year_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020'
    path_str=path_year_dir.split('/')
    output_file_name=path_str[-1]+'-'+path_str[-2]
    ani.ani_year_MP4(path_year_dir, path_output_dir, output_file_name+'.mp4', fps)


years=['2020','2021']
fps=15

for ii in range(len(years)):
    path_year_dir=path_input_dir+'/'+years[ii]
    year_GIF(path_year_dir, path_output_dir, fps)


ani.ani_year_MP4(path_input_dir, path_output_dir, output_file_name=path_output_dir+'/Dokdo.gif', fps=20)