# coding:utf-8
'''
nie
2016/3/31
'''

from numpy import *
import numpy
import copy
from random import *


def im2col_mean_removal(InImg, patchsize=[7, 7], stride=[1, 1], remove_mean=True):
    image_shape = InImg.shape
    if len(image_shape) == 3:
        rows, cols, z = image_shape
    else:
        InImg = InImg.reshape(InImg.shape[0], InImg.shape[1], 1)
        rows, cols, z = InImg.shape

    im_rows = len(range(0, rows - patchsize[0] + 1, stride[0]))
    im_cols = len(range(0, cols - patchsize[1] + 1, stride[1]))
    im = zeros((patchsize[0] * patchsize[1], im_rows * im_cols * z))
    idx = 0
    for chl in range(z):
        for i in range(0, rows - patchsize[0] + 1, stride[0]):
            for j in range(0, cols - patchsize[1] + 1, stride[1]):
                iim = InImg[i:(i + patchsize[0]), j:(j + patchsize[1]), chl]
                iim = iim.reshape(patchsize[0] * patchsize[1], 1)
                iim = asarray(iim)
                im[:, idx] = iim.T
                # im.append(iim)
                idx += 1
    im = im.reshape(patchsize[0] * patchsize[1], im_rows * im_cols * z)
    if remove_mean:
        im_mean = mean(im, axis=0)
        im = im - im_mean
    return im


def PCA_output(InImg, InImgIdx, patchsize, NumFilters, V):
    ImgZ = len(InImg)
    mag = (patchsize - 1) / 2
    OutImg = []
    cnt = 0

    for i in range(ImgZ):
        ImgX, ImgY, NumChls = InImg[i].shape
        img = zeros((ImgX + patchsize - 1, ImgY + patchsize - 1, NumChls))
        img[mag:(mag + InImg[i].shape[0]), mag:(mag + InImg[i].shape[1]), :] = InImg[i]
        im = im2col_mean_removal(img, [patchsize, patchsize], remove_mean=True)
        for j in range(NumFilters):
            OutImg.append(V[:, j].T.dot(im).reshape(ImgX, ImgY, 1))
            cnt += 1

    OutImgIdx = kron(InImgIdx, ones((1, NumFilters))[0])
    return OutImg, OutImgIdx


def PCA_FilterBank(InImg, patchsize, NumFilter):
    ImgZ = len(InImg)
    MaxSamples = 100000
    NumRSamples = min(ImgZ, MaxSamples)
    # RandIdx = range(ImgZ)
    # shuffle(RandIdx)
    # RandIdx = RandIdx[:NumRSamples]
    RandIdx = range(ImgZ)
    listInd = list(RandIdx)
    shuffle(listInd)
    RandIdx = listInd[:NumRSamples]

    NumChls = InImg[0].shape[2]
    Rx = zeros((NumChls * pow(patchsize, 2), NumChls * pow(patchsize, 2)), dtype="float32")

    for i in RandIdx:
        im = im2col_mean_removal(InImg[i], [patchsize, patchsize], remove_mean=True)
        Rx = Rx + im.dot(im.T)
    Rx = Rx / (NumRSamples * im.shape[1])
    D, E = linalg.eig(mat(Rx))
    Ind = argsort(D)
    Ind = Ind[:-(NumFilter + 1):-1]
    V = E[:, Ind]
    V = asarray(V)
    return V


def PCANet_FeaExt(PCANet, InImg, V):
    # assert(len(PCANet.NumFilters)!=PCANet.NumStages,'Length(PCANet.NumFilters)~=PCANet.NumStages')
    assert len(PCANet.NumFilters) == PCANet.NumStages, 'Length(PCANet.NumFilters)~=PCANet.NumStages'

    NumImg = len(InImg)
    OutImg = copy.deepcopy(InImg)
    ImgIdx = range(0, NumImg, 1)
    for stage in range(PCANet.NumStages):
        OutImg, ImgIdx = PCA_output(OutImg, ImgIdx, PCANet.PatchSize[stage], PCANet.NumFilters[stage], V[stage])
    f, BlkIdx = HashingHist(PCANet, ImgIdx, OutImg)
    return f, BlkIdx


def PCANet_train(InImg, PCANet, IdxExt):
    assert len(PCANet.NumFilters) == PCANet.NumStages, 'Length(PCANet.NumFilters)~=PCANet.NumStages'

    NumImg = len(InImg)  # 总的训练图片数目
    V = []
    OutImg = copy.deepcopy(InImg)
    # print(type(OutImg))
    ImgIdx = range(0, NumImg, 1)

    for stage in range(PCANet.NumStages):
        print('Computing PCA filter bank and its outputs at stage %d ...' % stage)
        V.append(PCA_FilterBank(OutImg, PCANet.PatchSize[stage], PCANet.NumFilters[stage]))
        if stage != PCANet.NumStages - 1:
            OutImg, ImgIdx = PCA_output(OutImg, ImgIdx, PCANet.PatchSize[stage], PCANet.NumFilters[stage], V[stage])
    print("stage")
    if IdxExt == 1:
        f = []
        for idx in range(NumImg):
            if mod(idx, 100) == 0:
                print('Extracting PCANet feasture of the  %d th training sample...' % idx)
            OutImgIndex, = where(array(ImgIdx) == idx)
            OutImg_tmp = []
            for i in OutImgIndex:
                OutImg_tmp.append(OutImg[i])
            OutImg_i, ImgIdx_i = PCA_output(OutImg_tmp, zeros((len(OutImgIndex), 1), dtype='int'), PCANet.PatchSize[-1],
                                            PCANet.NumFilters[-1], V[-1])
            f_idx, BlkIdx = HashingHist(PCANet, ImgIdx_i, OutImg_i)
            f.append(f_idx)
    else:
        f = []
        BlkIdx = []
    return f, V, BlkIdx


# def HashingHist(ImgIdx,OutImg):
#    NumImg=max(ImgIdx)
#   f=[]
#  map_weights=[pow(2,i) for i in range(NumFilters[-1]-1,-1,-1)]

#    for Idx in range(NumImg):
#       Idx_span=

def HashingHist(PCANet, img_idx, out_img):
    num_images = int(numpy.max(img_idx))
    map_weights = 2 ** array(range(PCANet.NumFilters[-1] - 1, -1, -1))
    patch_step = (1 - PCANet.BlkOverLapRatio) * array(PCANet.HistBlockSize)
    patch_step = [int(round(n, 0)) for n in patch_step]

    f = []
    bins = []
    histsize = 2 ** PCANet.NumFilters[-1]
    for idx in range(num_images + 1):
        Idx_span = where(array(img_idx) == idx)
        Idx_span = Idx_span[0]
        NumOs = len(Idx_span) // PCANet.NumFilters[-1]
        for i in range(NumOs):
            T = zeros(out_img[0].shape)
            for j in range(PCANet.NumFilters[-1]):
                signmap = sign(out_img[Idx_span[PCANet.NumFilters[-1] * i + j]])
                signmap[signmap <= 0] = 0
                T += map_weights[j] * (signmap)
            TT = im2col_mean_removal(T, PCANet.HistBlockSize, patch_step, remove_mean=False)
            bins = TT.shape[1]
            for k in range(bins):
                bhisttemp, binstemp = histogram(TT[:, k], range(histsize + 1))
                bhisttemp = asarray(bhisttemp)
                f.append(bhisttemp * (histsize / sum(bhisttemp)))
    f = array(f).reshape(1, bins * NumOs * histsize)[0]
    blkIdx = kron(ones((1, NumOs))[0], kron(array(range(bins)), ones((1, histsize))[0]))
    return f, blkIdx
