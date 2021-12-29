'''
animation library
'''

def ani_month_GIF(path_input_dir, path_output_dir, output_file_name, fps):
    '''
    path_input_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020/07'
    path_output_dir = 'D:/20_Product/animaition'
    output_file_name='202007-ACRI_L4-CHL-MULTI_4KM_GLO-DT.gif'
    fps= 5
    '''
    import os
    import imageio
    from PIL import Image
    path = [f"{path_input_dir}/{i}" for i in os.listdir(path_input_dir)]
    paths = [Image.open(i) for i in path]
    imageio.mimsave(path_output_dir+'/'+output_file_name, paths, fps=fps)

def ani_month_MP4(path_input_dir, path_output_dir, output_file_name, fps):
    '''
    path_input_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020/07'
    path_output_dir = 'D:/20_Product/animation'
    output_file_name='202007-ACRI_L4-CHL-MULTI_4KM_GLO-DT.mp4'
    fps= 5
    '''
    import cv2
    import os

    paths = [f"{path_input_dir}/{i}" for i in os.listdir(path_input_dir)]

    frame_array = []
    for ii in range(len(paths)):
        img = cv2.imread(paths[ii])
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
    out = cv2.VideoWriter(path_output_dir+'/'+output_file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def ani_year_MP4(path_input_dir, path_output_dir, output_file_name, fps):
    '''
    path_input_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP/2020'
    path_output_dir = 'D:/20_Product/animation'
    output_file_name='2020-ACRI_L4-CHL-MULTI_4KM_GLO-DT.mp4'
    fps= 5
    '''
    import cv2
    import os
    from Lib.lib_os import recursive_file

    paths = recursive_file(path_input_dir,'*.png')

    frame_array = []
    for ii in range(len(paths)):
        # ii = 0
        img = cv2.imread(paths[ii])
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
    out = cv2.VideoWriter(path_output_dir+'/'+output_file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

