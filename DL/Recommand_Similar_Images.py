'''
특정 이미지들을 입력받아,
재고에 있는 이미지 중
1차로 파일명이 동일한지 확인한다. 파일명이 동일한 것이 없으면 종료된다.
2차로 파일명과 동일한 파일들 중 유사도가 높은 순서대로 정렬한다.

'''


'''
This notebook represents a prototypical python 3 implementation for a without safeguards, without asserts, with fixed paramterized model. For larger data sets it is advised to store intermediate results, e.g. as pickle files.

Theory https://towardsdatascience.com/effortlessly-recommending-similar-images-b65aff6aabfb

# Rescaling

We assume to have a folder "originalImages" in the working directory. It shall contain jpg images.
As we will employ resnet18 using PyTorch, we need to resize the images to normalized 224x224 images
In a first step they are resized and stored in a different folder inputImagesCNN
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
## os.environ['KMP_DUPLICATE_LIB_OK']='True' # 다음 에러 발생 막기 위함
## OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import shutil

# needed input dimensions for the CNN
inputDim = (224,224)

### 경로선택 ###
## 테스트이미지
# path_main_dir='E:/Dokdo_DB/dokdo_image_recommendation/marinetest'
### 전체이미지
path_main_dir='E:/Dokdo_DB/독도사진전체'
### 홈페이지 업로드
# path_main_dir='E:/Dokdo_DB/독도종합정보시스템_업로드사진'

## 서브폴더 생성
inputDir = path_main_dir+'/'+"Original"
inputDirCNN = path_main_dir+'/'+"inputfigCNN"

os.makedirs(inputDirCNN, exist_ok = True)

transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
error_files=[]

extensions=['bmp','BMP','tif', 'TIF', 'jpg', 'JPG', 'png', 'PNG', 'gif', 'GIF', 'jpeg', 'JPEG',
                                            'tiff', 'TIFF']

## 오류유형에 따른 폴더 생성

os.makedirs(path_main_dir+'/error_extension', exist_ok=True)
os.makedirs(path_main_dir+'/error_read', exist_ok=True)
os.makedirs(path_main_dir+'/error_conversion', exist_ok=True)
os.makedirs(path_main_dir+'/error_dimension', exist_ok=True)
os.makedirs(path_main_dir+'/error_band', exist_ok=True)


file_list=os.listdir(inputDir)
### 필터링과 학습자료로 변환
for ii in tqdm(range(len(file_list))):
    # ii=0

    ### 지정한 확장자가 아니면 필터링
    if file_list[ii].split('.')[-1] not in extensions:
        shutil.move(inputDir+'/'+file_list[ii],path_main_dir+'/'+'error_extension')
        continue

    ### 확장자에 따른 이미지 읽기
    # if file_list[ii].split('.')[-1] != 'tif': ## TIF가 아니면 matplotlib(plt)을 이용함
    #     try:
    #         I= plt.imread(os.path.join(inputDir, file_list[ii]))
    #     except:
    #         shutil.move(inputDir + '/' + file_list[ii], path_main_dir + '/' + 'error_read')
    #         continue
    # else: ## TIF가 아니면 pillow(Image)을 이용함
    try:
        I = Image.open(os.path.join(inputDir, file_list[ii]))
    except:
        I.close()
        shutil.move(inputDir + '/' + file_list[ii], path_main_dir + '/' + 'error_read')
        continue
    try:
        Im = np.array(I)
    except: # Array변환 중에 오류나면 필터링
        I.close()
        shutil.move(inputDir+'/'+file_list[ii],path_main_dir+'/'+'error_conversion')
        continue

    ### 이미지 사이즈 안맞으면 오류파일로 지정
    if len(Im.shape)!=3: ## 3차원 구조 아니면 필터링
        I.close()
        shutil.move(inputDir+'/'+file_list[ii],path_main_dir+'/'+'error_dimension')
        continue
    elif min(Im.shape) != 3: ## 가장 작은 면적(밴드수)가 3이 아니면 필터링
        I.close()
        shutil.move(inputDir + '/' + file_list[ii], path_main_dir + '/' + 'error_band')
        continue

    # ### Array이미지를 pillow 이미지로 변한
    # try:
    #     I = Image.fromarray(I)
    # except:
    #     shutil.move(inputDir + '/' + file_list[ii], path_main_dir + '/' + 'error_conversion')
    #     continue

    # I = Image.open(os.path.join(inputDir, file_list[ii])).convert('RGB') # 이미지를 읽어 RGB로 변환
    try:
        newI = transformationForCNNInput(I) # CNN에 맞게 inputDim 크기로 변환
    except:
        I.close()
        shutil.move(inputDir + '/' + file_list[ii], path_main_dir + '/' + 'error_read')
        continue

    try:
        # copy the rotation information metadata from original image and save, else your transformed images may be rotated
        exif = I.info['exif']
        newI.save(os.path.join(inputDirCNN, file_list[ii]), exif=exif)
    except:
        newI.save(os.path.join(inputDirCNN, file_list[ii]))

    newI.close()
    I.close()


'''
Creating the similarity matrix with Resnet18
Let us first calculate the feature vectors with resnet18 on a CPU. The input is normalized to the ImageNet mean values/standard deviation.
'''

import torch
from torchvision import models
from torchvision import transforms


# for this prototype we use no gpu, cuda= False and as model resnet18 to obtain feature vectors

class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512

        return cnnModel, layer


# generate vectors for all the images in the set
img2vec = Img2VecResnet18()

allVectors = {}
print("Converting images to feature vectors:")
import matplotlib.pyplot as plt

inputfigCNN=path_main_dir+'/'+"inputfigCNN"
file_list=os.listdir(inputfigCNN)
error_files=[]


## 요청시 마다 벡터생성하면 응답시간이 늦으므로 일정주기로 vector를 업데이트하여 저장해야 함
for ii in tqdm(range(len(file_list))):
    I= Image.open(os.path.join(path_main_dir+'/'+"inputfigCNN", file_list[ii]))
    vec = img2vec.getVec(I)
    allVectors[file_list[ii]] = vec
    I.close()

### 벡터를 저장
print("Saving allVectors into allVectors.pkl:")
import pickle
path_allVectors=path_main_dir+'/allVectors.pkl'
allV=open(path_allVectors,'wb')
pickle.dump(allVectors, allV)
allV.close()

### 저장된 벡터를 읽음
print("Loading allVectors from allVectors.pkl:")
allV = open(path_allVectors, "rb")
allVectors = pickle.load(allV)
'''
Cosine similarity
Calculate for all vectors the cosine similarity to the other vectors. Note that this matrix may become huge, hence infefficient, with many thousands of images
'''

# now let us define a function that calculates the cosine similarity entries in the similarity matrix
import pandas as pd
import numpy as np


### 유사한 벡터를 찾음
def getSimilarityMatrix(vectors):
    ## vectors= allVectors
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / (
                (np.linalg.norm(v, axis=0).reshape(-1, 1)) * ((np.linalg.norm(v, axis=0).reshape(-1, 1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns=keys, index=keys)

    return matrix

### 여기서도 시간 엄청 걸림
similarityMatrix = getSimilarityMatrix(allVectors)


'''
Prepare top-k lists
Now that the similarity matrix is fully available, the last step is to sort the values per item and store the top similar entries in another data structure
'''

from numpy.testing import assert_almost_equal
import pickle
import pandas as pd

k = 5  # the number of top similar images to be stored

similarNames = pd.DataFrame(index=similarityMatrix.index, columns=range(k))
similarValues = pd.DataFrame(index=similarityMatrix.index, columns=range(k))

for j in tqdm(range(similarityMatrix.shape[0])):
    kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending=False).head(k)
    similarNames.iloc[j, :] = list(kSimilar.index)
    similarValues.iloc[j, :] = kSimilar.values

## 유사도 정보 저장
similarNames.to_pickle(path_main_dir+"/similarNames.pkl")
similarValues.to_pickle(path_main_dir+"/similarValues.pkl")

## 유사도 정보 읽기
similarNames = pickle.load(open(path_main_dir+"/similarNames.pkl", "rb"))
similarValues = pickle.load(open(path_main_dir+"/similarValues.pkl", "rb"))

similarNames.to_csv(path_main_dir+'/similarNames.csv')

np.nanmin(similarValues)

'''
# Get and visualize similar images for four example inputs
'''

import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.pyplot as plt

# take three examples from the provided image set and plot
# inputImages = ["독도파노라마사진 (11).jpg", "camper0.jpg", "buildings0.jpg", "donkey0.jpg"]

def setAxes(ax, image, query=False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize=12)
    else:
        ax.set_xlabel("Similarity {1:1.3f}\n{0}".format(image, value), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

def getSimilarImages(image, simNames, simVals):
    if image in set(simNames.index):
        imgs = list(simNames.loc[image, :])
        vals = list(simVals.loc[image, :])
        if image in imgs:
            assert_almost_equal(max(vals), 1, decimal=5)
            imgs.remove(image)
            vals.remove(max(vals))
        return imgs, vals
    else:
        print("'{}' Unknown image".format(image))


def plotSimilarImages(image, simiarNames, similarValues):
    simImages, simValues = getSimilarImages(image, similarNames, similarValues)
    fig = plt.figure(figsize=(10, 4))

    # now plot the  most simliar images
    for j in range(0, numCol * numRow):
        ax = []
        if j == 0:
            img = Image.open(os.path.join(inputDir, image))
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image, query=True)
        else:
            img = Image.open(os.path.join(inputDir, simImages[j - 1]))
            ax.append(fig.add_subplot(numRow, numCol, j + 1))
            setAxes(ax[-1], '', value=simValues[j - 1])
            # setAxes(ax[-1], simImages[j - 1], value=simValues[j - 1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()
    plt.show()

ss=np.random.randint(0,886,5)

top5unsimilar=similarValues[1][np.argsort(similarValues[1])][ss]
top5similar=similarValues[1][np.argsort(similarValues[1])][ss]

dir_test= path_main_dir+'/Test'
inputImages = list(top5similar.index)

numCol = 5
numRow = 1
## 이미지 파일명이 한글일 필요가 있음
for image in inputImages:
    plotSimilarImages(image, similarNames, similarValues)