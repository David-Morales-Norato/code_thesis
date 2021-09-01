import tensorflow as tf
from .Muestreos import *
import numpy as np
import cv2

class FSI_Initial(tf.keras.layers.Layer):
    def __init__(self, p, float_dtype, complex_dtype, **kwargs):
        super(FSI_Initial, self).__init__(**kwargs)
        self.p = p
        self.complex_dtype = complex_dtype
        self.float_dtype = float_dtype
        

    def build(self, input_shape):
        super(FSI_Initial, self).build(input_shape)
        self.S = input_shape
        self.M = tf.constant(self.S[1] * self.S[2] * self.S[3], dtype=self.float_dtype)
        self.R = tf.cast(tf.math.ceil(tf.math.divide(self.M, self.p)), dtype=self.float_dtype)

        Z0_abs = tf.random.normal(shape=(1,1, self.S[1], self.S[2]), mean=0.0, stddev=1)
        Z0_angle= tf.random.normal(shape=(1, 1,self.S[1], self.S[2]), mean=0.0, stddev=(0.5*np.pi**(1/2)))
        Z0_abs = tf.cast(Z0_abs,self.complex_dtype)
        Z0_angle = tf.cast(Z0_angle,self.complex_dtype)
        Z0 = tf.multiply(Z0_abs, tf.math.exp(tf.multiply(Z0_angle, tf.constant(1j,self.complex_dtype))))
        self.Z0 = tf.math.divide(Z0, tf.norm(Z0, ord='fro',axis=(2,3), keepdims=True))

        

    def call(self, input):

        Y = input
        
        # INitializations
        Y = tf.cast(Y, dtype=self.float_dtype); Y = tf.transpose(Y,perm=[0,3,1,2])# (self.S[0], self.L, self.S[1], self.S[2])
        Z = tf.cast(self.Z0,self.complex_dtype)
        div = tf.cast(tf.math.multiply(self.M,self.R),self.complex_dtype)

        # GET YTR
        S = tf.shape(Y)
        Y_S = tf.math.divide(Y,self.S[1])
        y_s = tf.reshape(Y_S, (S[0], S[1]*S[2]*S[3])) # Vectoriza

        y_s = tf.sort(y_s, axis = 1, direction='DESCENDING')
        aux  = tf.gather(y_s, indices=tf.cast(self.R-1, tf.int64), axis=1)
        threshold = tf.reshape(aux, (S[0], 1, 1, 1))
        Ytr = tf.cast(Y_S>=threshold, dtype=Y_S.dtype) 
        return Ytr, self.Z0

class FSI_cell(tf.keras.layers.Layer):
    def __init__(self, p, k_size, tipo_muestreo, kappa, float_dtype,complex_dtype, **kwargs):
        super(FSI_cell, self).__init__(**kwargs)
        self.p = p
        self.k_size = k_size
        self.kappa = kappa
        self.tipo_muestreo = tipo_muestreo
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype

        k = cv2.getGaussianKernel(k_size,1)
        self.kernel = tf.constant(np.dot(k, k.T), shape=( k_size, k_size), dtype=self.float_dtype)

        
        # self.conv_op_real = lambda z: tf.nn.conv2d(z, self.kernel, 1, "SAME", data_format='NHWC')
        # self.conv_op_imag = lambda z: tf.nn.conv2d(z, self.kernel, 1, "SAME", data_format='NHWC')
        self.conv_real = tf.keras.layers.Conv2D(1, self.k_size, padding="same", use_bias=False, activation = None, kernel_initializer=tf.keras.initializers.Constant(self.kernel), trainable=False,name="FILTRO_ABS_INITIALIZATION")
        self.conv_imag = tf.keras.layers.Conv2D(1, self.k_size, padding="same", use_bias=False, activation = None, kernel_initializer=tf.keras.initializers.Constant(self.kernel), trainable=False,name="FILTRO_ANG_INITIALIZATION")
        

    def build(self, input_shape):
        super(FSI_cell, self).build(input_shape[0])
        self.S = input_shape[0]
        if self.tipo_muestreo =="FRAN":
          self.A = lambda y, mask: A_Fran(y, mask)
          self.AT = lambda y, mask: AT_Fran(y, mask)
        elif self.tipo_muestreo == "ASM":
          self.A = lambda y, mask,sftf: A_ASM_LAB(y, mask, sftf, self.kappa)
          self.AT = lambda y,mask,sftf: AT_ASM_LAB(tf.cast(y, self.complex_dtype), mask, sftf)
        elif self.tipo_muestreo == "FRESNELL":
          # X = np.arange(-self.S[1]//2, self.S[1]//2)
          # Y = np.arange(-self.S[2]//2, self.S[2]//2)
          # X, Y = np.meshgrid(X,Y)
          # Q_base = (X**2 + Y**2)*(self.dx**2)
          # Q1 = np.exp(-1j*(np.pi/self.wave_length/self.distance)*Q_base)
          # Q2 = np.exp(-1j*(np.pi*self.wave_length*self.distance)*(Q_base/(self.S[2]*self.dx)**2))
          # self.Q1 = tf.broadcast_to(tf.convert_to_tensor(Q1, dtype=self.complex_dtype), self.Masks.shape)
          # self.Q2 = tf.broadcast_to(tf.convert_to_tensor(Q2, dtype=self.complex_dtype), self.Masks.shape)
          self.A = lambda y, mask,Q: A_FRESNELL(y, mask, Q, self.kappa)
          self.AT = lambda y, mask,Q: AT_FRESNELL(tf.cast(y, self.complex_dtype), mask, Q)
        else:
          raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

    def call(self, input):


        Ytr = tf.cast(input[0], self.float_dtype)
        Z = input[1]
        self.Masks = input[2]
        self.var_de_interes = input[3]

        Z = self.AT(tf.multiply(Ytr, self.A(Z, self.Masks,self.var_de_interes)), self.Masks,self.var_de_interes)
        Z = tf.math.divide(Z,(self.S[3]**2)*(self.S[1]**2)*(self.S[2]**2)*self.p)
        Z = tf.expand_dims(Z,-1)

        z_real = tf.math.real(Z)
        z_imag = tf.math.imag(Z)
        
        Z_real_filt = self.conv_real(z_real) - self.conv_imag(z_imag)
        Z_imag_filt = self.conv_real(z_imag) + self.conv_imag(z_real)
        
        Z = tf.complex(Z_real_filt, Z_imag_filt)
        Z = tf.transpose(Z, [0, 3, 1, 2])
        Z = tf.math.divide(Z, tf.norm(Z, ord='fro', axis=(2,3), keepdims=True))
        return Z


class BackPropagationLayer(tf.keras.layers.Layer):
  def __init__(self, tipo_muestreo, complex_dtype, **kwargs):
      super(BackPropagationLayer, self).__init__(**kwargs)  
      self.tipo_muestreo = tipo_muestreo  
      self.complex_dtype = complex_dtype

  def build(self, input_shape):
      super(BackPropagationLayer, self).build(input_shape)
      if self.tipo_muestreo =="FRAN":
        self.AT = lambda y, mask: AT_Fran(y, mask)
      elif self.tipo_muestreo == "ASM":
        self.AT = lambda y,mask,sftf: AT_ASM_LAB(tf.cast(y, self.complex_dtype), mask, sftf)
      elif self.tipo_muestreo == "FRESNELL":
        # X = np.arange(-self.S[1]//2, self.S[1]//2)
        # Y = np.arange(-self.S[2]//2, self.S[2]//2)
        # X, Y = np.meshgrid(X,Y)
        # Q_base = (X**2 + Y**2)*(self.dx**2)
        # Q1 = np.exp(-1j*(np.pi/self.wave_length/self.distance)*Q_base)
        # Q2 = np.exp(-1j*(np.pi*self.wave_length*self.distance)*(Q_base/(self.S[2]*self.dx)**2))
        # self.Q1 = tf.broadcast_to(tf.convert_to_tensor(Q1, dtype=self.complex_dtype), self.Masks.shape)
        # self.Q2 = tf.broadcast_to(tf.convert_to_tensor(Q2, dtype=self.complex_dtype), self.Masks.shape)
        self.AT = lambda y,mask,Q : AT_FRESNELL(tf.cast(y, self.complex_dtype), mask, Q)

      else:
        raise Exception("Tipo muestreo: " + self.tipo_muestreo + " inválido")

  def call(self, input):

      #self.Masks = tf.cast(input[1],self.complex_dtype)
      Y = tf.cast(input[0], self.complex_dtype)
      self.Masks = input[1]
      self.var_de_interes = input[2]

      Y = tf.transpose(Y,perm=[0,3,1,2])
      Z0 = self.AT(Y,self.Masks,self.var_de_interes)
      
      Z0 = tf.expand_dims(Z0, -1)
      back_real = tf.math.real(Z0); 
      back_imag = tf.math.imag(Z0); 
      Z = tf.concat([back_real, back_imag], axis = -1)
      return Z
