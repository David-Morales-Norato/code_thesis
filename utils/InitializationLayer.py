from tensorflow.keras.layers import Layer
import tensorflow as tf
from Muestreos import *
import numpy as np
import cv2
from loss_and_metrics import Between
import os
from scipy.io import savemat, loadmat

const_j = tf.constant(1j, dtype=tf.complex128)

def complex_polar(x_abs, x_angle):
  return tf.multiply(x_abs, tf.math.exp(tf.multiply(x_angle, const_j)))

class FSI_Initial(Layer):
    def __init__(self, p = 6, name="FSI_initial"):
        super(FSI_Initial, self).__init__(name=name)
        self.p = p
        

    def build(self, input_shape, k_size=5):
        super(FSI_Initial, self).build(input_shape)
        self.S = input_shape
        self.M = tf.constant(self.S[1] * self.S[2] * self.S[3], dtype=tf.float64)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=tf.float64)

        Z0_abs = tf.random.normal(shape=(self.S[0],1, self.S[1], self.S[2]), mean=0.0, stddev=1)
        Z0_angle= tf.random.normal(shape=(self.S[0], 1,self.S[1], self.S[2]), mean=0.0, stddev=(0.5*np.pi**(1/2)))
        #Z0_abs = tf.random.uniform(shape=(self.S[0],1, self.S[1], self.S[2]), maxval=1)
        #Z0_angle = tf.random.uniform(shape=(self.S[0],1, self.S[1], self.S[2]), maxval=2)
        Z0_abs = tf.cast(Z0_abs, tf.complex128)
        Z0_angle = tf.cast(Z0_angle, tf.complex128)
        Z0 = complex_polar(Z0_abs, Z0_angle)#tf.multiply(, tf.math.exp(tf.multiply(Z0_angle, tf.constant(1j, dtype=tf.complex128))))
        self.Z0 = tf.math.divide(Z0, tf.norm(Z0, ord='fro',axis=(2,3), keepdims=True))

        

    def call(self, input):

        Y = input
        
        # INitializations
        Y = tf.cast(Y, dtype=tf.float64); Y = tf.transpose(Y,perm=[0,3,1,2])# (self.S[0], self.L, self.S[1], self.S[2])
        Z = tf.cast(self.Z0, tf.complex128)
        div = tf.cast(tf.math.multiply(self.M,self.R), tf.complex128)

        # GET YTR
        S = tf.shape(Y)
        Y_S = tf.math.divide(Y,self.S[1])
        y_s = tf.reshape(Y_S, (S[0], S[1]*S[2]*S[3])) # Vectoriza

        y_s = tf.sort(y_s, axis = 1, direction='DESCENDING')
        aux  = tf.gather(y_s, indices=tf.cast(self.R-1, tf.int32), axis=1)
        threshold = tf.reshape(aux, (S[0], 1, 1, 1))
        Ytr = tf.cast(Y_S>=threshold, dtype=Y_S.dtype) 
        return Ytr, self.Z0

class FSI_cell(Layer):
    def __init__(self, mascara_lab = None, p = 4, k_size = 5, name="Learnableinit", sftf_folder = "..", tipo_muestreo = "ASM", archivo_sftf = "SFTF_LAB_simul.mat",  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", kernel_file = "kernel_fsi.mat", kappa = 1e-3):
        super(FSI_cell, self).__init__(name=name)
        self.k_size = k_size
        self.p = p
        self.kappa = tf.cast(kappa, tf.float64)
        self.Masks = mascara_lab
        self.tipo_muestreo = tipo_muestreo
        self.sftf_folder = sftf_folder
        self.archivo_sftf = archivo_sftf
        self.archivo_sftf_solver = archivo_sftf_solver

        # nombre_mat = os.path.join(self.sftf_folder, kernel_file)
        # kernel_real = tf.constant(loadmat(nombre_mat)["kernel_real"], dtype=tf.float64)
        # kernel_imag = tf.constant(loadmat(nombre_mat)["kernel_imag"], dtype=tf.float64)
        # kernel_real = tf.keras.initializers.Constant(kernel_real)
        # kernel_imag = tf.keras.initializers.Constant(kernel_imag)
        k = cv2.getGaussianKernel(k_size,1)
        self.kernel = tf.constant(np.dot(k, k.T), shape=(k_size, k_size,1, 1), dtype=tf.float64)
        #self.kernel = tf.keras.initializers.Constant(self.kernel)
        #print("KENREL real SHAPE", kernel_real.shape)
        #print("KENREL imag SHAPE", kernel_imag.shape)
        
        self.conv_op_real = lambda z: tf.nn.conv2d(z, self.kernel, 1, "SAME", data_format='NCHW')
        self.conv_op_imag = lambda z: tf.nn.conv2d(z, self.kernel, 1, "SAME", data_format='NCHW')
        #self.conv_real = tf.keras.layers.Conv2D(1, self.k_size, padding="same", use_bias=False, activation = None, trainable=False,name="FILTRO_ABS_INITIALIZATION", data_format='channels_first', kernel_initializer = self.kernel)
        #self.conv_imag = tf.keras.layers.Conv2D(1, self.k_size, padding="same", use_bias=False, activation = None, trainable=False,name="FILTRO_ANG_INITIALIZATION", data_format='channels_first', kernel_initializer = self.kernel)
        

    def build(self, input_shape, k_size=5):
        super(FSI_cell, self).build(input_shape[0])
        self.S = input_shape[0]


        if self.tipo_muestreo =="FRAN":
          self.A = lambda y: A_Fran(y, self.Masks)
          self.AT = lambda y: AT_Fran(y, self.Masks)
        elif self.tipo_muestreo == "ASM":
          nombre_mat = os.path.join(self.sftf_folder, self.archivo_sftf)
          SFTF_ang = loadmat(nombre_mat)["ang_SFTF"]
          SFTF_ang = tf.convert_to_tensor(SFTF_ang, dtype=tf.complex128)
          SFTF_ang = tf.expand_dims(tf.expand_dims(SFTF_ang,0),0)
          SFTF = complex_polar(tf.ones(tf.shape(SFTF_ang), tf.complex128), SFTF_ang)
          self.A = lambda y, mask: A_ASM_LAB(y, mask, SFTF, self.kappa)

          nombre_mat = os.path.join(self.sftf_folder, self.archivo_sftf_solver)
          SFTF_ang = loadmat(nombre_mat)["ang_SFTF"]
          SFTF_ang = tf.convert_to_tensor(SFTF_ang, dtype=tf.complex128)
          SFTF_ang = tf.expand_dims(tf.expand_dims(SFTF_ang,0),0)
          SFTF = complex_polar(tf.ones(tf.shape(SFTF_ang), tf.complex128), SFTF_ang)
          self.AT = lambda y,mask: AT_ASM_LAB(y, mask, SFTF)

        else:
          raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

        # self.A = lambda y: A_Fran(y, tf.reshape(self.Masks, (1,*self.Masks.shape)))
        # self.AT = lambda y: AT_Fran(y,tf.reshape(self.Masks, (1,*self.Masks.shape)))       

    def call(self, input):


        Ytr = tf.cast(input[0], tf.float64)
        Z = input[1]
        self.Masks = input[2]

        Z = self.AT(tf.multiply(Ytr, self.A(Z, self.Masks)), self.Masks)
        Z = tf.math.divide(Z,(self.S[3]**2)*(self.S[1]**2)*(self.S[2]**2)*self.p)
        Z = tf.expand_dims(Z,1)
        z_real = tf.math.real(Z)
        z_imag = tf.math.imag(Z)
        
        Z_real_filt = self.conv_op_real(z_real) - self.conv_op_imag(z_imag)
        Z_imag_filt = self.conv_op_real(z_imag) + self.conv_op_imag(z_real)
        Z = tf.complex(Z_real_filt, Z_imag_filt)
        Z = tf.math.divide(Z, tf.norm(Z, ord='fro', axis=(2,3), keepdims=True))

        return Z, self.Masks
