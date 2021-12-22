import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(v) == str:
                v = v.encode()
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in tqdm(range(nSamples)):
        imagePath = imagePathList[i]
        labelPath = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        with open(labelPath, 'r') as f:
            label = f.read().strip().split()
            label = ' '.join(label)
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print(imagePath, label)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    domain = "semantic"  # semantic or agnostic
    distorted = False  # distorted or not
    datasetPath = "./dataset/{}".format(domain)  # lmdb数据集存放路径，需自行设置
    
    items = os.listdir('./data/Corpus')  # 需要放入数据集的数据项名称列表，可自行设置
    imgPaths = [os.path.join("data", "Corpus", img, ("{}_distorted.jpg" if distorted else "{}.png").format(img)) for img in items]
    labelPaths = [os.path.join("data", "Corpus", img, "{}.{}".format(img, domain)) for img in items]

    createDataset(datasetPath, imgPaths, labelPaths)
