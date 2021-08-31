import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from Muestreos import *
from scipy.io import savemat, loadmat
import os

def set_phase_mask_R(L):
    def complementary(W):
        PhaseM = tf.math.exp(tf.multiply(tf.cast(W, dtype=tf.complex128), tf.constant(1j, dtype=tf.complex128)))
        #PhaseM = W
        E = tf.reduce_sum(tf.cast(tf.cast(tf.math.multiply(tf.math.conj(PhaseM), PhaseM), dtype=tf.complex128),dtype=tf.float64), axis=0)
        R1 = tf.reduce_sum(tf.square(E - tf.multiply(tf.ones((PhaseM.shape)), L)))
        R2 = tf.reduce_sum(tf.multiply(tf.multiply(tf.multiply(tf.square(W), tf.square(np.pi - W)), tf.square(-np.pi / 2 - W)), tf.square(np.pi / 2 - W)))
        #R2 = tf.reduce_sum(tf.multiply(tf.square(W), tf.square(1 - W)))
        return tf.multiply(tf.cast(R1, dtype=tf.float64) + tf.cast(R2, dtype=tf.float64), 1e-5)
    return complementary


class Muestreo(Layer):
    def __init__(self, snapshots, name="Muestreo", sftf_folder = "..", tipo_muestreo = "ASM", kappa = 1e-3, trainable=False, mascara_lab = None, archivo_sftf = "SFTF_LAB_simul.mat"):
        super(Muestreo, self).__init__(name=name)
        self.tipo_muestreo = tipo_muestreo
        self.L = snapshots
        self.trainable = trainable
        self.kappa = tf.cast(kappa, tf.float64)
        self.sftf_folder = sftf_folder
        self.mascara_lab = mascara_lab
        self.archivo_sftf = archivo_sftf

    def build(self, input_shape):
        super(Muestreo, self).build(input_shape)
        self.S = input_shape
        if (self.mascara_lab is None):
          Masks_ang = tf.random.uniform((1,self.L, *self.S[1:-1]), maxval=32/255)

          #Masks_ang = tf.random.uniform((self.S[1],self.S[2]), maxval=32/255)
          Masks_ang = Masks_ang*0.5*np.pi#-np.pi
          #   self.Masks_weights = self.add_weight(name='Masks', shape=[1,self.L, *self.S[1:-1]], initializer=tf.keras.initializers.Constant(Masks_ang), regularizer=set_phase_mask_R(self.L), trainable=self.trainable)
        else:
          self.mascara_lab = tf.cast(self.mascara_lab, dtype=tf.float64)
          Masks_ang = tf.expand_dims(tf.expand_dims(self.mascara_lab, 0),0)
          self.Masks_weights = self.add_weight(name='Masks', shape=[1,self.L, *self.S[1:-1]], initializer=tf.keras.initializers.Constant(Masks_ang), regularizer=set_phase_mask_R(self.L), trainable=self.trainable)
          
        self.Masks = tf.multiply(tf.ones(tf.shape(Masks_ang), dtype=tf.complex128), tf.math.exp(tf.multiply(tf.cast(Masks_ang,tf.complex128) , tf.constant(1j, dtype=tf.complex128))))
        if self.tipo_muestreo =="FRAN":
          self.A = lambda y: A_Fran(y, self.Masks)
        elif self.tipo_muestreo == "ASM":
          nombre_mat = os.path.join(self.sftf_folder, self.archivo_sftf)
          SFTF_ang = loadmat(nombre_mat)["ang_SFTF"]
          SFTF_ang = tf.convert_to_tensor(SFTF_ang, dtype=tf.complex128)
          SFTF_ang = tf.expand_dims(tf.expand_dims(SFTF_ang,0),0)
          SFTF = tf.multiply(tf.ones(tf.shape(SFTF_ang), dtype=tf.complex128), tf.math.exp(tf.multiply(SFTF_ang, tf.constant(1j, dtype=tf.complex128))))
          #SFTF = tf.ones(tf.shape(SFTF_ang), dtype=tf.complex128)
          self.A = lambda y: A_ASM_LAB(y, self.Masks, SFTF, self.kappa)
        else:
          raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

    def call(self, input):
        input = tf.cast(input, dtype=tf.float64)
        real, imag = tf.unstack(input, num=2, axis=3)
        Z = tf.complex(real, imag)
        Z = tf.expand_dims(Z,1)
        Y = self.A(Z)
        

        Y = tf.transpose(Y,perm=[0,2,3,1])  
        return Y, self.Masks
        
    def get_mask(self):
      return self.Masks


