import tensorflow as tf
import numpy as np
from .Muestreos import *
from scipy.io import savemat, loadmat
import os

class Muestreo(tf.keras.layers.Layer):
    def __init__(self, snapshots, wave_length ,dx ,distance, tipo_muestreo, kappa, mascara_lab, normalize_measurements, float_dtype, complex_dtype, trainable=False, **kwargs):
        super(Muestreo, self).__init__(**kwargs)
        self.tipo_muestreo = tipo_muestreo
        self.L = snapshots
        self.trainable = trainable
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype
        self.normalize_measurements = normalize_measurements
        self.kappa = tf.cast(kappa, dtype = self.float_dtype)
        self.mascara_lab = mascara_lab
        self.wave_length = wave_length
        self.dx = dx
        self.distance = distance

    def build(self, input_shape):
        super(Muestreo, self).build(input_shape)
        self.S = input_shape
        if (self.mascara_lab is None):
          Masks_ang = tf.random.uniform((1,self.L, *self.S[1:-1]))
          Masks_ang = Masks_ang*0.5*np.pi#-np.pi
        else:
          self.mascara_lab = tf.cast(self.mascara_lab, dtype=self.float_dtype)
          Masks_ang = tf.broadcast_to(self.mascara_lab, shape = [1,self.L, *self.S[1:-1]])
          self.Masks_weights = self.add_weight(name='Masks', shape=[1,self.L, *self.S[1:-1]], initializer=tf.keras.initializers.Constant(Masks_ang), trainable=self.trainable)
          
        self.Masks = tf.multiply(tf.ones(tf.shape(Masks_ang), dtype=self.complex_dtype), tf.math.exp(tf.multiply(tf.cast(Masks_ang,self.complex_dtype) , tf.constant(1j, dtype=self.complex_dtype))))
    
        if self.tipo_muestreo =="FRAN":
          self.A = lambda y: A_Fran(y, self.Masks)

        elif self.tipo_muestreo == "ASM":
          X = np.arange(-self.S[1]//2, self.S[1]//2)
          Y = np.arange(-self.S[2]//2, self.S[2]//2)
          X, Y = np.meshgrid(X,Y)
          U = 1 - (self.wave_length**2)*((X/(self.dx*self.S[1]))**2 + (Y/(self.dx*self.S[2]))**2);
          SFTF = np.exp(1j*2*np.pi/self.wave_length*self.distance*np.sqrt(U))
          SFTF[U<0]=0
          self.SFTF = tf.broadcast_to(tf.convert_to_tensor(SFTF, dtype=self.complex_dtype), self.Masks.shape)
          self.A = lambda y: A_ASM_LAB(y, self.Masks, self.SFTF, self.kappa)
          self.var_de_interes = self.SFTF

        elif self.tipo_muestreo == "FRESNELL":
          X = np.arange(-self.S[1]//2, self.S[1]//2)
          Y = np.arange(-self.S[2]//2, self.S[2]//2)
          X, Y = np.meshgrid(X,Y)
          Q_base = (X**2 + Y**2)
          Q1 = np.exp(-1j*(np.pi/self.wave_length/self.distance)*Q_base*(self.dx**2))
          Q2 = np.exp(-1j*(np.pi*self.wave_length*self.distance)*((Q_base)/((self.S[2]*self.dx)**2)))
          self.Q1 = tf.broadcast_to(tf.convert_to_tensor(Q1, dtype=self.complex_dtype), self.Masks.shape)
          self.Q2 = tf.broadcast_to(tf.convert_to_tensor(Q2, dtype=self.complex_dtype), self.Masks.shape)
          self.var_de_interes = (self.Q1, self.Q2)
          self.A = lambda y: A_FRESNELL(y, self.Masks, self.var_de_interes, self.kappa)
        else:
          raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

    def call(self, input):
        input = tf.cast(input, dtype=self.float_dtype)
        real, imag = tf.unstack(input, num=2, axis=3)
        Z = tf.complex(real, imag)
        Z = tf.expand_dims(Z,1)
        Y = self.A(Z)
        Y = tf.transpose(Y,perm=[0,2,3,1])  
        if(self.normalize_measurements):
          division = tf.math.reduce_max(Y, axis=[1,2], keepdims=True)
          Y = tf.math.divide(Y, division)
        return Y, self.Masks, self.var_de_interes
        
    def get_mask(self):
      return self.Masks
    def get_sftf(self):
      return self.SFTF

    def get_var_de_interes(self):
      return self.var_de_interes