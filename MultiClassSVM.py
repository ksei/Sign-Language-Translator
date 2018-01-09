import numpy as np
import cv2
import SVM
import os
from collections import defaultdict
import utils


class MCSVM(object):
    def __init__(self):
        self.w = {}
        self.b = {}

    def multiClassFit(self):

        features = defaultdict(list)
        w = {}
        b = {}

        clf = SVM.SVM()

        for root, dirs, files in os.walk('./Train Data'):
            for sample in files:
                filename = os.path.join(sample)
                path = 'Train Data/' + filename

                im = cv2.imread(path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                hog = utils.calc_HOG(im)
                features[filename[0]].append(hog.transpose(1, 0)[0])

        labels = list(features.keys())

        for i in range(0, len(labels)):
            for j in range(i + 1, len(labels)):
                print labels[i]+labels[j]

        for i in range(0, len(labels)):
            for j in range(i + 1, len(labels)):
                neg = np.full((len(features[labels[i]])), -1.0)
                pos = np.full(( len(features[labels[j]])), 1.0)
                tempLabel = np.concatenate((neg, pos), axis=0)
                tempFeatures = np.concatenate((np.array(features[labels[i]]), np.array(features[labels[j]])), axis=0)

                clf.fit(tempFeatures, tempLabel);
                self.w[labels[i] + labels[j]] = clf.w
                self.b[labels[i] + labels[j]] = clf.b

        np.savez('storedSVM\\model', w=self.w, b=self.b)


    def multiClassPredict(self, HOG):

        votes = defaultdict(int)
        data = np.load('storedSVM/model.npz')
        w = data['w'].item()
        b = data['b'].item()

        labels =  list(w.keys())

        svm = SVM.SVM()


        for i in range(0, len(labels)):

            svm.w = w[labels[i]]
            svm.b = b[labels[i]]



            if svm.predict(HOG) < 0 :
                votes[labels[i][0]] = votes[labels[i][0]] + 1
            else:
                votes[labels[i][1]] = votes[labels[i][1]] + 1




        return max(votes, key=votes.get)



'''
msvm = MCSVM();

#msvm.multiClassFit();

#data = np.load('storedSVM/model.npz')
#data1 = np.load('storedSVM/test.npz')

im = cv2.imread("Train Data/E1.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
hog = calc_HOG(im).transpose(1,0)[0];

print msvm.multiClassPredict(hog)
'''