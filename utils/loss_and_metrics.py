import tensorflow as tf
import numpy as np
from keras.constraints import Constraint
import keras.backend as K

class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):        
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


def Loss_Abs(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val = 1))
  #psnr = 50 - tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val = 1))
  return ssim + mse(y_true, y_pred)


def Loss_Ang(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val = 2*np.pi))
  #psnr = 50 - tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val = 2*np.pi))
  return ssim  + mse(y_true, y_pred)

def RE(y_true, y_pred):
  cons_j = tf.constant(1j, dtype=tf.complex64)

  abs_real = y_true[:,:,0]
  ang_real = y_true[:,:,1]

  abs_pred = y_pred[:,:,0]
  ang_pred = y_pred[:,:,1]

  x = tf.multiply(tf.cast(abs_real, dtype=tf.complex64), tf.math.exp(tf.multiply(tf.cast(ang_real, dtype=tf.complex64), cons_j)))
  z = tf.multiply(tf.cast(abs_pred, dtype=tf.complex64), tf.math.exp(tf.multiply(tf.cast(ang_pred, dtype=tf.complex64), cons_j)))

  x_T = tf.transpose(x)


  mul = tf.linalg.matmul(x_T,z)
  trace = tf.broadcast_to(tf.linalg.trace(mul), x.shape)
  angle = tf.cast(tf.math.angle(trace), tf.complex64)
  exp_xz =  tf.math.exp(tf.multiply(-angle, cons_j))
  return tf.math.real(tf.math.divide(tf.norm( x - tf.linalg.matmul(exp_xz,z),2 ), tf.norm(x,2 )))



## CLASIFICATION METRICS





