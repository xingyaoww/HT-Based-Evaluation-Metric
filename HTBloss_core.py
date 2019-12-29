'''
HTBloss_core Function

Call `HTB_Loss(y_pred, y_true, lane_status)`,
It will returns a value denotes HTBloss,
it will return 1. if HTBloss does not exist.
'''
import cv2
import numpy as np
import skimage
import skimage.measure
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

class HTB_Loss:
    # Normalization the Params
    @staticmethod
    def minmaxScale(data):
        # Accept 2 col
        ret = np.copy(data)

        def normalizeCol(col):
            minval = np.min(col)
            maxval = np.max(col)
            interval = maxval - minval
            return (col - minval) / interval, (minval, interval)

        ret[:, 0], r1 = normalizeCol(ret[:, 0])
        ret[:, 1], r2 = normalizeCol(ret[:, 1])
        return ret, (r1, r2)

    @staticmethod
    def reverseMinMaxScale(data, reverseinfo):
        ret = np.copy(data)
        col1info, col2info = reverseinfo

        def reverseNormalizeCol(col, colreverseinfo):
            minval, interval = colreverseinfo
            return (col * interval) + minval

        ret[:, 0] = reverseNormalizeCol(ret[:, 0], col1info)
        ret[:, 1] = reverseNormalizeCol(ret[:, 1], col2info)
        return ret

    @staticmethod
    def poolingratio(factor, ratio=294 / 820):
        return (round(ratio * factor), factor)

    @staticmethod
    def maxpooling(img, blocksizex, blocksizey):
        return skimage.measure.block_reduce(img, (blocksizex, blocksizey), np.max)

    @staticmethod
    def img2binary(img):
        img = np.round(img)
        img[np.where(img > 0)] = 1
        img[np.where(img < 0)] = 0
        img = img.astype(np.uint8)
        return img

    # Hough Transform
    @staticmethod
    def getLanePara(lanes, HTthres=200):
        '''
        :param lanes: should be a list including labeled image OR predicted image.
                     value: [0, 1, 2, 3, 4] for different lane
        :return a list of HTparam [[rho, theta], ...]
        '''
        HTparams = []
        for i, lane in enumerate(lanes):
            #         lane = img2binary(lane)
            #         plt.imshow(lane)
            HTparam = cv2.HoughLines(lane, 1, np.pi / 180, HTthres)
            HTparams.append(HTparam)
        return HTparams




    @staticmethod
    def calHTBasedLoss(
            #     valsample=ValidateData[666],
            predictimg, labelimg, lanestatus,
            HTthres=50,
            poolsize=None,
            errorreturn=1.):
        '''
        REQUIRE: Input loaded predictImage(by pass original image into the model), LabelImage, and an array of lanestatus.
        EFFECTS: Return the calculated HTBased Loss for given input sample,
        return `None` if the HTBased Loss cannot be found for given input.
        '''
        '''
        :param poolsize: type int. indicates the blocksize of the pooling kernel
        '''
        if poolsize is None: poolsize = HTB_Loss.poolingratio(7)
        # Because lanestatus passed by the model is a float!
        n_lanes = len(np.where(lanestatus > 0.5)[0])

        # Transform from 3D to 2D
        labelimg = HTB_Loss.img2binary(np.sum(labelimg, axis=2))
        predictimg = HTB_Loss.img2binary(np.sum(predictimg, axis=2))

        # Max pooling if necessary
        if poolsize:
            labelimg = HTB_Loss.maxpooling(labelimg, poolsize[0], poolsize[1])
            predictimg = HTB_Loss.maxpooling(predictimg, poolsize[0], poolsize[1])

        labelParams = HTB_Loss.getLanePara([labelimg], HTthres)
        predictParams = HTB_Loss.getLanePara([predictimg], HTthres)
        if labelParams[0] is None or predictParams[0] is None:
            #         print("## LANE NOT DETECTED ##")
            return errorreturn
        labelParams = np.squeeze(labelParams[0])
        predictParams = np.squeeze(predictParams[0])

        try:
            std_labelParams, std_labelrev = HTB_Loss.minmaxScale(labelParams)
            std_predictParams, std_predictrev = HTB_Loss.minmaxScale(predictParams)
            # K-Mean
            labelKmeanPred = KMeans(n_clusters=n_lanes).fit_predict(std_labelParams)

            # Random oversampling K-Mean Result
            ros = RandomOverSampler(random_state=0)
            ros_std_labelParams, ros_labelKmeanPred = ros.fit_resample(
                std_labelParams, labelKmeanPred)

            # K-NN trained by ros oversampling K-Mean Result
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(ros_std_labelParams, ros_labelKmeanPred)
            predictKNNPred = neigh.predict(std_predictParams)

            # Calculate Mean for each Lane Class
            Means = np.zeros((n_lanes, 2, 2))
            for i in range(n_lanes):
                labelindex = np.where(labelKmeanPred == i)
                predictindex = np.where(predictKNNPred == i)
                if std_labelParams[labelindex].size == 0 or std_predictParams[predictindex].size == 0:
                    return errorreturn
                Means[i, 0] = np.mean(std_labelParams[labelindex], axis=0)
                Means[i, 1] = np.mean(std_predictParams[predictindex], axis=0)

            if np.isnan(Means).any(): return errorreturn
            # Output HTBased Error
            HTBloss = np.square(Means[:, 0, :] - Means[:, 1, :]).mean()

            # Check if HTBloss is valid
            if np.isnan(HTBloss):
                return errorreturn
        except:
            return errorreturn
        return HTBloss

    @staticmethod
    def calculate(y_pred, y_true, lane_status):
        '''
        A wrapper for calHTBloss()
        A mini-batch is passed in
        :param y_pred:
        :param y_true:
        :param lane_status:
        :return:
        '''
        ret = np.zeros((len(y_pred), 1), dtype=np.float32)
        for i in range(len(y_pred)):
            ret[i] = HTB_Loss.calHTBasedLoss(y_pred[i], y_true[i], lane_status[i])
        # print("Before Return ")
        return ret
