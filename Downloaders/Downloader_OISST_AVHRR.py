import requests
from bs4 import BeautifulSoup
import numpy as np

path_output_dir='D:/01_Model/OISST_onlyAVHRR'

def get_url_paths(yyyy,mm, ext='', params={}):
    '''
    yyyy='2021'
    mm='09'
    ext = 'nc'
    '''

    url = 'https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/' + yyyy + mm
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    urls = [url +'/'+ node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return urls

def download(url, path_output_dir):
    '''

    '''
    import requests
    # ii=0
    response = requests.get(url, stream=True)
    open(path_output_dir+'/'+url.split('/')[-1], 'wb').write(response.content)
    print('Downloaded :'+ url.split('/')[-1])


years=np.arange(2018,2021,1).astype(str)
months=['{0:02d}'.format(month) for month in np.arange(1,13,1)]

for ii in range(len(years)):
    for jj in range(len(months)):
        urls = get_url_paths(years[ii],months[jj], ext='nc', params={})
        for kk in range(len(urls)):
            download(urls[kk], path_output_dir)



