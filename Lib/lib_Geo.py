import numpy as np


def deg2gyro(deg):
    # deg=-45
    return np.remainder(360 + 90 - deg, 360)

def gyro2deg(gyro):
    # gyro=45
    return np.remainder(360 + 90 - gyro, 360)

def uv2gyrospd(uo,vo):
    # uo=ext_dataset['uo']
    # vo=ext_dataset['vo']
    # uo=2
    # vo=1
    # uo= np.array(DF_dict['uo']).astype(float)
    # vo= np.array(DF_dict['vo']).astype(float)
    gyro=deg2gyro(np.arctan2(vo.astype(float), uo.astype(float)) * (180 / np.pi))
    spd=np.sqrt(vo**2+uo**2)
    return gyro, spd

def gyrospd2uv(gyro,spd):
    # gyro = df['WindDir(deg)']
    # spd =  df['WindSpd(m/s)']
    # gyro=np.array(30)
    # spd=np.array(10)
    cond=~np.isnan(gyro.astype(float)) & ~np.isnan(np.array(spd))
    gyro = gyro.astype(float)[cond]
    spd = spd[cond]

    deg=gyro2deg(gyro)
    uo=np.cos(np.deg2rad(deg))*spd
    vo=np.sin(np.deg2rad(deg))*spd
    return uo, vo

def find_nearst_idx(mesh_lon, mesh_lat, target_lon, target_lat):
    '''
    meshgrid의 경위도에서 내가 원하는 경위도와 가장 가까운 위치의 index 반환
    target_lon = 131.8666  # Dokdo
    target_lat = 37.23972  # Dokdo
    '''

    # 유클리드 거리 계산
    dist=np.sqrt((mesh_lon-target_lon)**2+(mesh_lat-target_lat)**2)
    idx=np.where(dist==dist.min())
    return idx

def idxArray4match(mesh_lon,mesh_lat,target_meshlon, target_meshlat):
    '''
    mesh_lon/lat에 target_meshlon/lat에 위치한 데이터를 입력하는 방법으로
    최근접거리를 적용함

    target_meshlon=n_SLA_lon
    target_meshlat=n_SLA_lat
    '''
    import Lib.lib_GOCI1 as G1
    r,c=mesh_lon.shape
    id_x=np.zeros((r,c))
    id_x[:]=np.nan
    id_y=id_x.copy()

    for ii in range(r):
        for jj in range(c):
           y, x= find_nearst_idx(mesh_lon[ii,jj],mesh_lat[ii,jj],target_meshlon, target_meshlat)
           if len(y) > 1 or len(x) >1: # 만약 최근접이 2개 이상 나오게 되면,
               y, x = min(y), min(x) # 최소값으로 선택함
           id_y[ii,jj], id_x[ii,jj]= int(y), int(x)

    return id_y, id_x


def matchedArray(mesh_lon,mesh_lat,target_meshlon, target_meshlat, target_z):
    '''
    VBD 경위도좌표계에 맞추어, CHL, SST, SLA의 값을 입력한다
    이때 방식은 Nearest Neighbor

    입력자료 예)
    mesh_lon = mesh_lon
    mesh_lat = mesh_lat
    target_meshlon = n_CHL_lon
    target_meshlat = n_CHL_lat
    target_z = n_CHL[ii,:,:]
    '''


    y,x=idxArray4match(mesh_lon,mesh_lat,target_meshlon, target_meshlat)
    y,x = y.astype(int),x.astype(int) # 실수르 정수로 변환해야 indexing이 가능함

    r,c=y.shape

    z=np.zeros((r,c)) # 빈공간만들고
    z[:]=np.nan # nan으로 채우기

    ## 2중 for문을 통해 cell 하나하나의 값을 찾음
    for ii in range(r):
        for jj in range(c):
            z[ii,jj]=target_z[y[ii,jj],x[ii,jj]]

    return z