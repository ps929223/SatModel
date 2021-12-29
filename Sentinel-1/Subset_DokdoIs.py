import subprocess as sp
import os
import numpy as np

path_s1_input_dir='E:/02_Satellite/S2/Dokdo'
path_s1_output_dir='E:/02_Satellite/S2/Dokdo/SNAP'
path_graph='E:/02_Satellite/S2/Dokdo/graph/subset_DokdoIs.xml'

file_list=np.array(os.listdir(path_s1_input_dir))
file_list=file_list[np.array(['.zip' in name for name in file_list])]

for ii in range(len(file_list)):
     s1_input_name = file_list[ii]
     print(file_list[ii])
     s1_output_name = 'Subset_S2L2A_DokdoIs_' + s1_input_name.split('_')[2] + '.dim'
     cmd='gpt '+ path_graph+ ' ' +\
          '-Pfile='+path_s1_input_dir+'/'+s1_input_name+' '+\
          '-Ptarget='+path_s1_output_dir+'/'+s1_output_name
     sp.call(cmd)


