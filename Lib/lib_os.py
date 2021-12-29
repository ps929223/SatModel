def recursive_file(target_dir,pattern):
    '''
    특정 디렉토리 하부폴더 안에 있는 모든 파일의 경로를 반환
    # pattern = "*.png"
    '''
    import os
    from glob import glob

    files = []

    # target_dir='D:/20_Product/CMEMS/GLO-BIO-MULTI-L4-CHL-4KM-INTP'
    for dir, _, _ in os.walk(target_dir):
        files.extend(glob(os.path.join(dir, pattern),recursive=True))
    return files
