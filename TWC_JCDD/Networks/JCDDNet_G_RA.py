import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, constraints
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda, Activation
from tensorflow.keras.backend import mean, var, max, abs, square, sqrt
from SystemParams import LDPC_para, Train_para, MIMO_para

LDPC_N = int(LDPC_para['LDPC_N'])
# Gamma_v = int(LDPC_para['Gamma_v'])
# Gamma_c = int(LDPC_para['Gamma_c'])
A = np.array(LDPC_para['A'], dtype='float32')
Ashape = np.shape(A)
A_T = tf.transpose(A, perm=[1, 0])
Lambda_A = np.array(LDPC_para['Lambda_A'], dtype='float32')
theta = np.array(LDPC_para['theta'], dtype='float32')
L_num = int(Train_para['L_num'])
Nr = int(MIMO_para['Nr'])
Nt = int(MIMO_para['Nt'])
qAry = int(MIMO_para['qAry'])
Ld = int(MIMO_para['Ld'])
Lp = int(MIMO_para['Lp'])

class JCDD_G_layer(layers.Layer):
    def __init__(self):
        super(JCDD_G_layer, self).__init__()

    def build(self,input_shape):
        if qAry == 2:
            self.miu = self.add_weight(name='miu',shape=(1,1),initializer=keras.initializers.Constant(value=0.6),trainable=True)
            self.alpha = self.add_weight(name='alpha',shape=(1,1),initializer=keras.initializers.Constant(value=11),trainable=True)
        elif qAry == 4:
            self.miu = self.add_weight(name='miu',shape=(1,1),initializer=keras.initializers.Constant(value=0.2),trainable=True)
            self.alpha1 = self.add_weight(name='alpha1',shape=(1,1),initializer=keras.initializers.Constant(value=2.8),trainable=True)
            self.alpha2 = self.add_weight(name='alpha2',shape=(1,1),initializer=keras.initializers.Constant(value=2.8),trainable=True)
        self.factor = self.add_weight(name='factor',shape=(1,1),initializer=keras.initializers.Constant(value=1),trainable=True)
        self.relax = self.add_weight(name='relax',shape=(1,1),initializer=keras.initializers.Constant(value=1),trainable=True)
        self.acc = self.add_weight(name='acc',shape=(1,1),initializer=keras.initializers.Constant(value=0),trainable=True)

    def call(self, inputs):

        [sigma2_I,Y,YYp,real_Xp,imag_Xp,lambda_W,F,u,z,lambda1,acc_new] = inputs
        z_old = z
        lambda1_old = lambda1
        acc_old = acc_new

        if qAry == 2:
            fb = 4*lambda_W
            Ftemp = tf.transpose(F,[0,2,1])
            res = tf.reshape(Ftemp,[-1,Ld,2,Nt])
            res = tf.transpose(res,[0,1,3,2])
            res = tf.reshape(res,[-1,2*Nt*Ld,1])
            fb_res = 2*lambda_W - 2*np.sqrt(2)*res
            q = self.miu*tf.matmul(A_T, theta - z - lambda1) + fb_res - self.alpha
            w = 1/(self.miu*Lambda_A + fb - 2*self.alpha)
            u = tf.multiply(q,w)
            u = tf.nn.relu(u) - tf.nn.relu(u-1)
            u_order = tf.reshape(u,[-1,Ld,Nt,qAry])
            u_order = tf.transpose(u_order,[0,2,1,3])
            real_X = 1/np.sqrt(2)*(1-2*u_order[:,:,:,0:1])
            imag_X = 1/np.sqrt(2)*(1-2*u_order[:,:,:,1:2])
        elif qAry == 4:
            u_temp = tf.reshape(u,[-1,np.int(LDPC_N/4),4])
            u1 = tf.reshape(u_temp[:,:,0:2],[-1,np.int(LDPC_N/2),1])
            u2 = tf.reshape(u_temp[:,:,2:4],[-1,np.int(LDPC_N/2),1])
            Ftemp = tf.transpose(F,[0,2,1])
            res = tf.reshape(Ftemp,[-1,Ld,2,Nt])
            res = tf.transpose(res,[0,1,3,2])
            res = tf.reshape(res,[-1,2*Nt*Ld,1])

            A_T_temp = tf.reshape(A_T,[np.int(LDPC_N/4),4,Ashape[0]])
            A_T1 = tf.reshape(A_T_temp[:,0:2,:],[np.int(LDPC_N/2),Ashape[0]])
            A_T2 = tf.reshape(A_T_temp[:,2:4,:],[np.int(LDPC_N/2),Ashape[0]])

            Lambda_A_temp = tf.reshape(Lambda_A,[np.int(LDPC_N/4),4])
            Lambda_A1 = tf.reshape(Lambda_A_temp[:,0:2],[np.int(LDPC_N/2),1])
            Lambda_A2 = tf.reshape(Lambda_A_temp[:,2:4],[np.int(LDPC_N/2),1])

            fb1 = 8/10*lambda_W*tf.multiply(1+2*u2,1+2*u2)
            fb_res1 = 4/10*lambda_W*tf.multiply(1+2*u2,1+2*u2)- 4/np.sqrt(10)*tf.multiply(1+2*u2,res)
            q1 = self.miu*tf.matmul(A_T1, theta - z - lambda1) + fb_res1 - self.alpha1
            w1 = 1/(self.miu*Lambda_A1 + fb1 - 2*self.alpha1)
            u1 = tf.multiply(q1,w1)
            u1 = tf.nn.relu(u1) - tf.nn.relu(u1-1)

            fb2 = 8/10*lambda_W*tf.multiply(1-2*u1,1-2*u1)
            fb_res2 = -4/10*lambda_W*tf.multiply(1-2*u1,1-2*u1)+ 4/np.sqrt(10)*tf.multiply(1-2*u1,res)
            q2 = self.miu*tf.matmul(A_T2, theta - z - lambda1) + fb_res2 - self.alpha2
            w2 = 1/(self.miu*Lambda_A2 + fb2 - 2*self.alpha2)
            u2 = tf.multiply(q2,w2)
            u2 = tf.nn.relu(u2) - tf.nn.relu(u2-1)

            u = tf.concat([tf.reshape(u1,[-1,np.int(LDPC_N/4),2]),tf.reshape(u2,[-1,np.int(LDPC_N/4),2])],2)
            u= tf.reshape(u,[-1,LDPC_N,1])
            u_order = tf.reshape(u,[-1,Ld,Nt,qAry])
            u_order = tf.transpose(u_order,[0,2,1,3])
            real_X = 1/np.sqrt(10)*tf.multiply(1-2*u_order[:,:,:,0:1],1+2*u_order[:,:,:,2:3])
            imag_X = 1/np.sqrt(10)*tf.multiply(1-2*u_order[:,:,:,1:2],1+2*u_order[:,:,:,3:4])

        real_X = tf.squeeze(real_X,[3])
        imag_X = tf.squeeze(imag_X,[3])
        X = tf.concat([real_X,imag_X],1)
        real_XXp = tf.concat([real_X,real_Xp],2)
        imag_XXp = tf.concat([imag_X,imag_Xp],2)
        XXp = tf.concat([tf.concat([real_XXp,imag_XXp],2),tf.concat([-imag_XXp,real_XXp],2)],1)
        R = tf.matmul(XXp,XXp,transpose_b=True) + sigma2_I
        inv_R = tf.linalg.inv(R)
        W_temp = tf.matmul(tf.matmul(inv_R,XXp),YYp,transpose_b=True)
        real_W = W_temp[:,0:Nt,:]
        imag_W = W_temp[:,Nt:2*Nt,:]
        W =  tf.transpose(tf.concat([tf.concat([real_W,imag_W],2),tf.concat([-imag_W,real_W],2)],1),[0,2,1])
        lambda_W = self.factor*lambda_W
        F1 = tf.matmul(W,Y,transpose_a=True)
        F2 = tf.multiply(lambda_W,X)
        WTW = tf.matmul(W,W,transpose_a=True)
        F3 = tf.matmul(WTW,X)
        F = F1 + F2 - F3

        # step 2
        ztemp = theta - self.relax*tf.matmul(A, u) - (1-self.relax)*(theta - z_old) - lambda1
        z = ztemp
        z = tf.nn.relu(z)

        # step 3
        lambda1 = -ztemp + z

        # acc
        z = z + self.acc*(z-z_old)
        lambda1 = lambda1 + self.acc*(lambda1-lambda1_old)

        return [lambda_W,F,u,z,lambda1,acc_new]

def JCDD_G(Lnum) -> Model:

    z = tf.zeros([Ashape[0], 1])
    lambda1 = tf.zeros([Ashape[0], 1])
    acc = tf.ones([1, 1])

    u_len = np.int(LDPC_N/2/Nt)
    len_arr = u_len+2*(Ld+Lp)+Nt+2*Ld+Ld+1
    inpt = layers.Input(shape = (2*Nt, len_arr))
    u = inpt[:,:,:u_len]
    u = tf.transpose(u,[0,2,1])
    u = tf.reshape(u,[-1,LDPC_N,1])
    YYp = inpt[:,:,u_len:(u_len+2*(Ld+Lp))]
    YYp = tf.transpose(YYp,[0,1,2])
    real_Xp = inpt[:,:Nt,(u_len+2*(Ld+Lp)):(u_len+2*(Ld+Lp)+Nt)]
    real_Xp = tf.transpose(real_Xp,[0,1,2])
    imag_Xp = inpt[:,Nt:2*Nt,(u_len+2*(Ld+Lp)):(u_len+2*(Ld+Lp)+Nt)]
    imag_Xp = tf.transpose(imag_Xp,[0,1,2])
    real_Y = inpt[:,:,(u_len+2*(Ld+Lp)+Nt):(u_len+2*(Ld+Lp)+Nt+Ld)]
    imag_Y = inpt[:,:,(u_len+2*(Ld+Lp)+Nt+Ld):(u_len+2*(Ld+Lp)+Nt+2*Ld)]
    Y = tf.concat([real_Y,imag_Y],1)
    F = inpt[:,:,(u_len+2*(Ld+Lp)+Nt+2*Ld):(u_len+2*(Ld+Lp)+Nt+3*Ld)]
    F = tf.transpose(F,[0,1,2])
    lambda_W = inpt[:,0:1,(len_arr-1):len_arr]
    lambda_W = tf.transpose(lambda_W,[0,1,2])
    sigma2 = inpt[:,1:2,(len_arr-1):len_arr]
    sigma2 = tf.transpose(sigma2,[0,1,2])
    I_ = inpt[:,2:3,(len_arr-1):len_arr]
    I_ = tf.transpose(I_,[0,1,2])

    z = tf.matmul(z,I_)
    lambda1 = tf.matmul(lambda1,I_)
    sigma2_I = tf.reshape(tf.matmul(tf.ones([2*Nt,1]),sigma2),[-1,2*Nt])
    sigma2_I = tf.linalg.diag(sigma2_I)
    acc_new = acc
    for ll in range(Lnum):
        [lambda_W,F,u,z,lambda1,acc_new] = JCDD_G_layer()([sigma2_I,Y,YYp,real_Xp,imag_Xp,lambda_W,F,u,z,lambda1,acc_new])
    u_est = u[:,:LDPC_N,:] - 0.5
    u_est = tf.nn.tanh(200*u_est)
    out = tf.concat([u_est,u,z],1)
    model = Model(inputs=inpt, outputs=out, name='JCDDNetG')

    return model
