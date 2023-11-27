import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, constraints
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda, Activation
from tensorflow.keras.backend import mean, var, max, abs, square, sqrt
from SystemParams_mmWave import LDPC_para, Train_para, MIMO_para, DFT_para

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
real_Ur = np.array(DFT_para['real_Ur'], dtype='float32')
imag_Ur = np.array(DFT_para['imag_Ur'], dtype='float32')
real_Ut = np.array(DFT_para['real_Ut'], dtype='float32')
imag_Ut = np.array(DFT_para['imag_Ut'], dtype='float32')

Ut = tf.concat([real_Ut,imag_Ut],0)
real_UtT = tf.transpose(real_Ut,[1,0])
imag_UtT = -tf.transpose(imag_Ut,[1,0])
real_UrT = tf.transpose(real_Ur,[1,0])
imag_UrT = -tf.transpose(imag_Ur,[1,0])
UrT = tf.concat([tf.concat([real_UrT,-imag_UrT],1),tf.concat([imag_UrT,real_UrT],1)],0)
Ur = tf.concat([tf.concat([real_Ur,-imag_Ur],1),tf.concat([imag_Ur,real_Ur],1)],0)
UtT = tf.concat([tf.concat([real_UtT,-imag_UtT],1),tf.concat([imag_UtT,real_UtT],1)],0)
UtT2 = tf.concat([real_UtT,imag_UtT],0)

class JCDD_S_layer(layers.Layer):
    def __init__(self):
        super(JCDD_S_layer, self).__init__()

    def build(self,input_shape):
        if qAry == 2:
            self.rho = self.add_weight(name='rho',shape=(1,1),initializer=keras.initializers.Constant(value=0.2),trainable=True) #constraint=constraints.non_neg(),
            self.kappa = self.add_weight(name='kappa',shape=(1,1),initializer=keras.initializers.Constant(value=2.8),trainable=True)
        elif qAry == 4:
            self.rho = self.add_weight(name='rho',shape=(1,1),initializer=keras.initializers.Constant(value=0.1),trainable=True) #constraint=constraints.non_neg(),
            self.kappa1 = self.add_weight(name='kappa1',shape=(1,1),initializer=keras.initializers.Constant(value=1),trainable=True)
            self.kappa2 = self.add_weight(name='kappa2',shape=(1,1),initializer=keras.initializers.Constant(value=1),trainable=True)
        self.epsilon = self.add_weight(name='epsilon',shape=(1,1),initializer=keras.initializers.Constant(value=5),trainable=True)
        self.tao = self.add_weight(name='tao',shape=(1,1),initializer=keras.initializers.Constant(value=0.001),trainable=True)
        self.factor = self.add_weight(name='factor',shape=(1,1),initializer=keras.initializers.Constant(value=1),trainable=True)
        self.relax = self.add_weight(name='relax',shape=(1,1),initializer=keras.initializers.Constant(value=1),trainable=True)
        self.acc = self.add_weight(name='acc',shape=(1,1),initializer=keras.initializers.Constant(value=0),trainable=True)

    def call(self, inputs):

        [sigma2,X,XXp,Y,YYp,real_Xp,imag_Xp,H,lambda_W,u,z,lambda1,acc_new] = inputs
        z_old = z
        lambda1_old = lambda1
        acc_old = acc_new

        UrH_temp = tf.matmul(UrT,H)
        real_UrH = UrH_temp[:,:Nr,:]
        imag_UrH = UrH_temp[:,Nr:2*Nr,:]
        UrH = tf.concat([tf.concat([real_UrH,-imag_UrH],2),tf.concat([imag_UrH,real_UrH],2)],1)
        Hv_temp = tf.matmul(UrH,Ut)
        real_Hv = Hv_temp[:,:Nr,:]
        imag_Hv = Hv_temp[:,Nr:2*Nr,:]
        Hv = tf.concat([tf.concat([real_Hv,-imag_Hv],2),tf.concat([imag_Hv,real_Hv],2)],1)
        UtTXXp = tf.matmul(UtT,XXp)
        UrTYYp = tf.matmul(UrT,YYp)
        real_UtTXXp = UtTXXp[:,:Nt,:]
        imag_UtTXXp = UtTXXp[:,Nt:2*Nt,:]
        UtTXXpT = tf.concat([tf.transpose(real_UtTXXp,[0,2,1]),tf.transpose(-imag_UtTXXp,[0,2,1])],1)
        delta_temp = tf.matmul(Hv,tf.concat([real_UtTXXp,imag_UtTXXp],1)) - UrTYYp
        real_delta = delta_temp[:,:Nr,:]
        imag_delta = delta_temp[:,Nr:2*Nr,:]
        delta = tf.concat([tf.concat([real_delta,-imag_delta],2),tf.concat([imag_delta,real_delta],2)],1)
        Hvest = Hv_temp - self.tao*tf.matmul(delta,UtTXXpT)
        real_Hvest = Hvest[:,:Nr,:]
        imag_Hvest = Hvest[:,Nr:2*Nr,:]
        abs_Hvest = tf.sqrt(tf.multiply(real_Hvest,real_Hvest) + tf.multiply(imag_Hvest,imag_Hvest))
        abs_Hvest = tf.concat([abs_Hvest,abs_Hvest],1)
        max_relu = tf.nn.relu(abs_Hvest-self.tao*self.epsilon*sigma2)
        Hvest = tf.multiply(tf.multiply(Hvest,1/abs_Hvest),max_relu)
        UrHv_temp = tf.matmul(Ur,Hvest)
        real_UrHv = UrHv_temp[:,:Nr,:]
        imag_UrHv = UrHv_temp[:,Nr:2*Nr,:]
        UrHv = tf.concat([tf.concat([real_UrHv,-imag_UrHv],2),tf.concat([imag_UrHv,real_UrHv],2)],1)
        H = tf.matmul(UrHv,UtT2)
        real_H = H[:,0:Nr,:]
        imag_H = H[:,Nr:2*Nr,:]
        W =  tf.concat([tf.concat([real_H,-imag_H],2),tf.concat([imag_H,real_H],2)],1)
        lambda_W = self.factor*lambda_W
        F1 = tf.matmul(W,Y,transpose_a=True)
        F2 = tf.multiply(lambda_W,X)
        WTW = tf.matmul(W,W,transpose_a=True)
        F3 = tf.matmul(WTW,X)
        F = F1 + F2 - F3

        if qAry == 2:
            fb = 4*lambda_W
            Ftemp = tf.transpose(F,[0,2,1])
            res = tf.reshape(Ftemp,[-1,Ld,2,Nt])
            res = tf.transpose(res,[0,1,3,2])
            res = tf.reshape(res,[-1,2*Nt*Ld,1])
            fb_res = 2*lambda_W - 2*np.sqrt(2)*res
            q = self.rho*tf.matmul(A_T, theta - z - lambda1) + fb_res - self.kappa
            w = 1/(self.rho*Lambda_A + fb - 2*self.kappa)
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
            q1 = self.rho*tf.matmul(A_T1, theta - z - lambda1) + fb_res1 - self.kappa1
            w1 = 1/(self.rho*Lambda_A1 + fb1 - 2*self.kappa1)
            u1 = tf.multiply(q1,w1)
            u1 = tf.nn.relu(u1) - tf.nn.relu(u1-1)

            fb2 = 8/10*lambda_W*tf.multiply(1-2*u1,1-2*u1)
            fb_res2 = -4/10*lambda_W*tf.multiply(1-2*u1,1-2*u1)+ 4/np.sqrt(10)*tf.multiply(1-2*u1,res)
            q2 = self.rho*tf.matmul(A_T2, theta - z - lambda1) + fb_res2 - self.kappa2
            w2 = 1/(self.rho*Lambda_A2 + fb2 - 2*self.kappa2)
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
        XXp = tf.concat([real_XXp,imag_XXp],1)

        # step 2
        ztemp = theta - self.relax*tf.matmul(A, u) - (1-self.relax)*(theta - z_old) - lambda1
        z = ztemp
        z = tf.nn.relu(z)

        # step 3
        lambda1 = -ztemp + z

        # acc
        z = z + self.acc*(z-z_old)
        lambda1 = lambda1 + self.acc*(lambda1-lambda1_old)

        return [X,XXp,H,lambda_W,u,z,lambda1,acc_new]

def JCDD_S(Lnum) -> Model:

    z = tf.zeros([Ashape[0], 1])
    lambda1 = tf.zeros([Ashape[0], 1])
    acc = tf.ones([1, 1])

    u_len = np.int(LDPC_N/2/Nt)
    YYp_len = np.int((Ld+Lp)*Nr/Nt)
    Y_len = np.int(Ld*Nr/Nt)
    H_len = Nr
    len_arr = u_len+YYp_len+Lp+Y_len+Ld+Ld+Lp+H_len+1
    inpt = layers.Input(shape = (2*Nt, len_arr))
    u = inpt[:,:,:u_len]
    u = tf.transpose(u,[0,2,1])
    u = tf.reshape(u,[-1,LDPC_N,1])

    YYp = inpt[:,:,u_len:(u_len+YYp_len)]
    YYp = tf.transpose(YYp,[0,2,1])
    YYp = tf.reshape(YYp,[-1,Ld+Lp,2*Nr])
    YYp = tf.transpose(YYp,[0,2,1])

    real_Xp = inpt[:,:Nt,(u_len+YYp_len):(u_len+YYp_len+Nt)]
    real_Xp = tf.transpose(real_Xp,[0,1,2])
    imag_Xp = inpt[:,Nt:2*Nt,(u_len+YYp_len):(u_len+YYp_len+Nt)]
    imag_Xp = tf.transpose(imag_Xp,[0,1,2])

    Y = inpt[:,:,(u_len+YYp_len+Nt):(u_len+YYp_len+Nt+Y_len)]
    Y = tf.transpose(Y,[0,2,1])
    Y = tf.reshape(Y,[-1,Ld,2*Nr])
    Y = tf.transpose(Y,[0,2,1])

    X = inpt[:,:,(u_len+YYp_len+Nt+Y_len):(u_len+YYp_len+Nt+Y_len+Ld)]
    X = tf.transpose(X,[0,1,2])
    XXp = inpt[:,:,(u_len+YYp_len+Nt+Y_len+Ld):(u_len+YYp_len+Nt+Y_len+Ld+Lp+Ld)]
    XXp = tf.transpose(XXp,[0,1,2])

    H = inpt[:,:,(u_len+YYp_len+Nt+Y_len+Ld+Lp+Ld):(u_len+YYp_len+Nt+Y_len+Ld+Lp+Ld+H_len)]
    H = tf.transpose(H,[0,2,1])
    H = tf.reshape(H,[-1,Nt,2*Nr])
    H = tf.transpose(H,[0,2,1])

    lambda_W = inpt[:,0:1,(len_arr-1):len_arr]
    lambda_W = tf.transpose(lambda_W,[0,1,2])
    sigma2 = inpt[:,1:2,(len_arr-1):len_arr]
    sigma2 = tf.transpose(sigma2,[0,1,2])
    I_ = inpt[:,2:3,(len_arr-1):len_arr]
    I_ = tf.transpose(I_,[0,1,2])

    z = tf.matmul(z,I_)
    lambda1 = tf.matmul(lambda1,I_)
    acc_new = acc

    for ll in range(Lnum):
        [X,XXp,H,lambda_W,u,z,lambda1,acc_new] = JCDD_S_layer()([sigma2,X,XXp,Y,YYp,real_Xp,imag_Xp,H,lambda_W,u,z,lambda1,acc_new])

    u_est = u[:,:LDPC_N,:] - 0.5
    u_est = tf.nn.tanh(200*u_est)
    out = tf.concat([u_est,u,z],1)
    model = Model(inputs=inpt, outputs=out, name='JCDDNetS')

    return model
