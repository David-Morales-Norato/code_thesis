from tensorflow.keras.layers import Layer
from scipy.io import savemat, loadmat
from Muestreos import AT_ASM_LAB, AT_Fran
import tensorflow as tf
import numpy as np
import os


# class BackPropagationLayer(Layer):
#     def __init__(self, snapshots, name="BackPropagationLayer", sftf_folder = "..", tipo_muestreo = "ASM", trainable=False, mascara_lab = None, archivo_sftf_solver = "SFTF_LAB_simul.mat"):
#         super(BackPropagationLayer, self).__init__(name=name)
#         self.tipo_muestreo = tipo_muestreo
#         self.trainable = trainable
#         self.L = snapshots
#         self.sftf_folder = sftf_folder
#         self.mascara_lab = mascara_lab
#         self.archivo_sftf_solver = archivo_sftf_solver

#     def build(self, input_shape):
#         super(BackPropagationLayer, self).build(input_shape)
#         self.S = input_shape
#         self.mascara_lab = tf.cast(self.mascara_lab, dtype=tf.float32)
#         Masks_ang = tf.expand_dims(tf.expand_dims(self.mascara_lab, 0),0)
#         self.Masks_weights = self.add_weight(name='Masks', shape=[1,self.L, *self.S[1:-1]], initializer=tf.keras.initializers.Constant(Masks_ang), regularizer=set_phase_mask_R(self.L), trainable=self.trainable)
        
#         self.Masks = tf.multiply(tf.ones(tf.shape(Masks_ang), dtype=tf.complex64), tf.math.exp(tf.multiply(tf.cast(self.Masks_weights,tf.complex64) , tf.constant(1j, dtype=tf.complex64))))
#         if self.tipo_muestreo =="FRAN":
#           self.A = lambda y: A_Fran(y, self.Masks)
#         elif self.tipo_muestreo == "ASM":
#           nombre_mat = os.path.join(self.sftf_folder, self.archivo_sftf_solver)
#           SFTF_ang = loadmat(nombre_mat)["ang_SFTF"]
#           SFTF_ang = tf.convert_to_tensor(SFTF_ang, dtype=tf.complex64)
#           SFTF_ang = tf.expand_dims(tf.expand_dims(SFTF_ang,0),0)
#           SFTF = tf.multiply(tf.ones(tf.shape(SFTF_ang), dtype=tf.complex64), tf.math.exp(tf.multiply(SFTF_ang, tf.constant(1j, dtype=tf.complex64))))
#           self.AT = lambda y: AT_ASM_LAB(y, self.Masks, SFTF)
#         else:
#           raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

#     def call(self, input):
#         muestras = tf.cast(input, dtype=tf.complex64)
#         back = self.AT(Z)
#         return back
        
#     def get_mask(self):
#       return self.Masks

class BackPropagationLayer(Layer):
  def __init__(self, tipo_muestreo = "ASM", mascara_lab = None, sftf_folder = "..",  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", name="BackPropagationLayer"):
      super(BackPropagationLayer, self).__init__(name=name)  
      self.archivo_sftf_solver = archivo_sftf_solver
      self.mascara_lab = mascara_lab
      self.tipo_muestreo = tipo_muestreo  
      self.sftf_folder = sftf_folder

  def build(self, input_shape):
      super(BackPropagationLayer, self).build(input_shape)

      self.Masks = self.mascara_lab
      if self.tipo_muestreo =="FRAN":
        self.AT = lambda y: AT_Fran(y, self.Masks)
      elif self.tipo_muestreo == "ASM":
        nombre_mat = os.path.join(self.sftf_folder, self.archivo_sftf_solver)
        SFTF_ang = loadmat(nombre_mat)["ang_SFTF"]
        SFTF_ang = tf.convert_to_tensor(SFTF_ang, dtype=tf.complex128)
        SFTF_ang = tf.expand_dims(tf.expand_dims(SFTF_ang,0),0)
        SFTF = tf.multiply(tf.ones(tf.shape(SFTF_ang), dtype=tf.complex128), tf.math.exp(tf.multiply(SFTF_ang, tf.constant(1j, dtype=tf.complex128))))
        self.AT = lambda y: AT_ASM_LAB(y, self.Masks, SFTF)
      else:
        raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

  def call(self, input):

      #self.Masks = tf.cast(input[1], tf.complex128)
      Y = tf.cast(input[0], tf.complex128)
      self.Masks = input[1]

      Y = tf.transpose(Y,perm=[0,3,1,2])
      Z0 = self.AT(Y)
      
      Z0 = tf.expand_dims(Z0, -1)
      back_real = tf.math.real(Z0); 
      back_imag = tf.math.imag(Z0); 
      

      Z = tf.concat([back_real, back_imag], axis = -1)

      return Z

  def set_mask(self, Masks):
    self.Masks = Masks
