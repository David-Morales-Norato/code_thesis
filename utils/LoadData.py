import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import h5py

def preprocess_mnist(image,label, shape, num_clases):
  x = tf.image.resize(image, shape)
  mins = tf.math.reduce_min(x,axis=[1,2], keepdims=True)
  maxs = tf.math.reduce_max(x,axis=[1,2], keepdims=True)
  x = tf.math.divide(tf.math.subtract(x,mins),tf.math.subtract(maxs,mins))
  x = x*0.5*np.pi#-np.pi
  x_abs = tf.cast(tf.ones(tf.shape(x)), tf.complex64)
  x_angle = tf.cast(x, tf.complex64)
  x = tf.multiply(x_abs, tf.math.exp(tf.multiply(x_angle, tf.constant(1j, dtype=tf.complex64))))
  x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
  
  label = tf.one_hot(tf.cast(label, tf.int32), num_clases)
  return x,label

def preprocess_cifar(image,label, shape, num_clases):
  x = tf.math.reduce_mean(image, axis=-1, keepdims=True)
  x = tf.image.resize(x, shape)
  

  mins = tf.math.reduce_min(x,axis=[1,2], keepdims=True)
  maxs = tf.math.reduce_max(x,axis=[1,2], keepdims=True)
  x = tf.math.divide(tf.math.subtract(x,mins),tf.math.subtract(maxs,mins))

  x = x*0.5*np.pi#-np.pi
  x_abs = tf.cast(tf.ones(tf.shape(x)), tf.complex64)
  x_angle = tf.cast(x, tf.complex64)
  x = tf.multiply(x_abs, tf.math.exp(tf.multiply(x_angle, tf.constant(1j, dtype=tf.complex64))))
  x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)

  label = tf.one_hot(tf.cast(label, tf.int32), num_clases)
  return x,label

def get_datos(name_dataset = "mnist", batch_size=8, tam = (128,128), num_clases=10):

  train_images = tfds.load(name_dataset, split='train', as_supervised=True, batch_size=batch_size)
  test_images = tfds.load(name_dataset, split='test', as_supervised=True, batch_size=batch_size)

  if name_dataset == "cifar10":
    preprocess = preprocess_cifar
  else:
    preprocess = preprocess_mnist

  train_images = train_images.map(lambda x,y: preprocess(x,y, shape= tam, num_clases=num_clases))
  test_images = test_images.map(lambda x,y: preprocess(x,y, shape= tam, num_clases=num_clases))

  return train_images, test_images

