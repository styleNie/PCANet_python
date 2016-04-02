#coding:utf-8
'''
nie
2016/4/1
'''

import gzip
import os
import sys
import six.moves.cPickle as pickle
from numpy import *
import copy
import time
import pcanet
from sklearn import *
import warnings

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set

class PCANet(object):
    def __init__(self,NumStages=2,PatchSize=[7,7],NumFilters=[8,8],
                 HistBlockSize=[7,7],BlkOverLapRatio=0.5):
        self.NumStages=NumStages
        self.PatchSize=PatchSize
        self.NumFilters=NumFilters
        self.HistBlockSize=HistBlockSize
        self.BlkOverLapRatio=BlkOverLapRatio

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    test,valid,train=load_data('mnist.pkl.gz')
    train_x,train_y=train
    valid_x,valid_y=valid
    test_x,test_y=test

    train_x=train_x[:10];train_y=train_y[:10]
    test_x=test_x[:10];test_x=test_x[:10]
    
    rows=28
    cols=28

    train_x=train_x.reshape(train_x.shape[0],rows,cols,1)
    valid_x=valid_x.reshape(valid_x.shape[0],rows,cols,1)
    test_x=test_x.reshape(test_x.shape[0],rows,cols,1)
    
    print('train_x shape:%s'%list(train_x.shape))  # 10000
    print('valid_x shape:%s'%list(valid_x.shape))   # 10000
    print('test_x shape:%s'%list(test_x.shape))    # 50000

    TrnSize=10000
    ImgSize=28
    PCANet=PCANet()
    print('\n ====== PCANet Parameters ======= \n')
    print('NumStages= %d'%PCANet.NumStages)
    print('PatchSize=[%d, %d]'%(PCANet.PatchSize[0],PCANet.PatchSize[1]))
    print('NumFilters=[%d, %d]'%(PCANet.NumFilters[0],PCANet.NumFilters[1]))
    print('HistBlockSize=[%d, %d]'%(PCANet.HistBlockSize[0],PCANet.HistBlockSize[1]))
    print('BlkOverLapRatio= %f'%PCANet.BlkOverLapRatio)

    print('\n ====== PCANet Training ======= \n')
    start = time.time()
    ftrain,V,BlkIdx = pcanet.PCANet_train(train_x,PCANet,1)
    end= time.time()
    PCANet_TrnTime=end-start
    print('PCANet training time:%f'%PCANet_TrnTime)

    print('\n ====== Training Linear SVM Classifier ======= \n')
    start = time.time()
    classifier=svm.LinearSVC()
    svm_model= classifier.fit(ftrain,train_y)
    end= time.time()
    LinearSVM_TrnTime=end-start
    print('SVM classfier training time:%f'%LinearSVM_TrnTime)

    print('\n ====== PCANet Testing ======= \n')
    acc=0
    start = time.time()
    for i in range(len(test_x)):
        test_x_i=test_x[i].reshape((1,)+test_x[i].shape)
        ftest,BlkIdx = pcanet.PCANet_FeaExt(PCANet,test_x_i,V)
        pred_label=svm_model.predict(ftest)
        pred_true=test_y[i]
        if pred_label==pred_true:
            acc+=1
            
    end= time.time()
    Averaged_TimeperTest=(end-start)/len(test_x)
    Accuracy=float(acc)/len(test_x)
    ErRate=1-Accuracy
    #print('test accuracy %f'%(float(acc)/len(test_x)))

    print('===== Results of PCANet, followed by a linear SVM classifier =====\n')
    print('     PCANet training time:  %.2f secs.'%PCANet_TrnTime)
    print('     Linear SVM training time:  %.2f secs.'%LinearSVM_TrnTime)
    print('     Testing error rate:  %.2f%%'%(100*ErRate))
    print('     Average testing time  %.2f  secs per test sample. '%Averaged_TimeperTest);
        
