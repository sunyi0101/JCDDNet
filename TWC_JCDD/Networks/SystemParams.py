import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda, Activation
from tensorflow.keras.backend import mean, var, max, abs, square, sqrt
import numpy as np

LDPC_N = 288
LDPC_R = 0.5
Nr = 8
Nt = 4
qAry = 2
Ld = np.int(LDPC_N/Nt/qAry)
Lp = Nt
MIMO_para = {'Nr':Nr,'Nt':Nt,'qAry':qAry,'Ld':Ld,'Lp':Lp}

DFT = loadmat('E:\JDD_DataSet\DFT_Ur64Ut8.mat')
real_Ur = DFT['real_Ur']
imag_Ur = DFT['imag_Ur']
real_Ut = DFT['real_Ut']
imag_Ut = DFT['imag_Ut']
DFT_para = {'real_Ur':real_Ur,'imag_Ur':imag_Ur,'real_Ut':real_Ut,'imag_Ut':imag_Ut}

batchsize = 200
epoch = 200
L_num = 100

LDPC = loadmat('E:\JDD_DataSet\LDPC_144_288_A.mat')
A = LDPC['A']
Lambda_A = LDPC['lambda_A']
theta = LDPC['theta']
LDPC_para = {'LDPC_N':LDPC_N, 'A':A, 'Lambda_A':Lambda_A, 'theta':theta}
Train_para = {'batchsize':batchsize, 'epoch':epoch, 'L_num':L_num}

