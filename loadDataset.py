'''
This script load CULane dataset from ./CULane/list
'''

import keras
import math
import os
import cv2
import numpy as np

# dataset: input image, per-pixel label,
# four 0/1 numbers which indicate the existance of
#  four lane markings from left to right

defaultpath = os.path.join(os.path.join(os.getcwd(),"CULane"), "list")

def getTestDataPath(TRAINING_LIMIT=34680, listpath=defaultpath):
    '''

    :param TRAINING_LIMIT: Default get all - 34680
    :param listpath:
    :return:
    '''
    # Load CULane List file:
    # listpath = 'CULane/list'
    # TRAINING_LIMIT = 10
    dataset = list()
    with open(os.path.join(listpath, 'test.txt'), 'r') as f:
        for i in range(TRAINING_LIMIT):
            # Remove /n at the end of the line
            line = f.readline().split(' ')
            line[-1] = line[-1].replace('\n', '')
            line[0] = 'CULane'+line[0]
            dataset.append(line)
    return dataset

def getTrainDataPath(TRAINING_LIMIT=88880, listpath=defaultpath):
    '''
    :param TRAINING_LIMIT: MAX 88880 , Default get all
    :param listpath: the path of CULane/list/, generally is os.path.join(os.path.join(os.getcwd(),"CULane"), "list")
    :return: dataset [[traindatapath, labelpath, 0laneexist, 1laneexist, 2laneexist, 3laneexist]]
    '''
    # Load CULane List file:
    # listpath = 'CULane/list'
    # TRAINING_LIMIT = 10
    dataset = list()
    with open(os.path.join(listpath, 'train_gt.txt'), 'r') as f:
        for i in range(TRAINING_LIMIT):
            # Indicates Lane Existing Status
            # Remove /n at the end of the line
            line = f.readline().split(' ')
            line[-1] = line[-1].replace('\n', '')

            train_sample = 'CULane'+line[0]
            y_true = 'CULane'+line[1]
            lane_status = line[2:]
            dataset.append([train_sample, y_true, lane_status])

    return dataset


def getValidateDataPath(TRAINING_LIMIT=9675, listpath=defaultpath):
    '''
    :param TRAINING_LIMIT: Default get all - 9675
    :param listpath:
    :return: same to get TrainDataPath
    '''
    # Default set to load MAX validation data
    # MAX 9675
    dataset = list()
    with open(os.path.join(listpath, 'val_gt.txt'), 'r') as f:
        for i in range(TRAINING_LIMIT):
            # Remove /n at the end of the line
            line = f.readline().split(' ')
            line[-1] = line[-1].replace('\n', '')

            train_sample = 'CULane' + line[0]
            y_true = 'CULane' + line[1]
            lane_status = line[2:]
            dataset.append([train_sample, y_true, lane_status])

    # dataset = np.array(dataset)
    # trainset is [traindata, label]
    # valset = dataset[:, 0:2].tolist()
    # lane_status = dataset[:, 2:].tolist()
    # print(trainset, lane_status)
    return dataset



class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, batch_size=16, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        # Loaded samples counter
        self.counter = 0

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]

        X, y_true, lane_status = self.data_generation(batch_datas)

        # Uncomment below to Run with Custom Metric
        # ret = [np.array(X), np.array(lane_status)]
        ret = [np.array(X)]
        return ret, np.array(y_true)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []
        lane_status = []
        # 生成数据

        for i, data in enumerate(batch_datas):
            # data[0] -> training dataset
            # data[1] -> training labels

            img = cv2.imread(data[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to increase Efficiency
            img = cv2.resize(img, (820, 294), interpolation=cv2.INTER_CUBIC)
            images.append(img)

            img = cv2.imread(data[1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (820, 294), interpolation=cv2.INTER_CUBIC)
            labels.append(img)

            lane_status.append(np.array(data[2]).reshape((4, 1)))
        images = np.array(images)
        labels = np.array(labels)

        # Threshold the ground truth to Binary Image for classfication problem.
        from HTBloss_core import HTB_Loss
        labels = HTB_Loss.img2binary(labels)

        lane_status = np.array(lane_status)
        lane_status[np.where(lane_status == '1')] = 1
        lane_status[np.where(lane_status == '0')] = 0
        return images, labels, lane_status

if __name__ == '__main__':
    dataset = getTrainDataPath()
    print(dataset)

'''
Sample Call:

training_generator = loadDataset.DataGenerator(loadDataset.getTrainDataPath(TRAINING_LIMIT=Trainning_Limit))

'''