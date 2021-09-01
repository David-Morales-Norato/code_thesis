import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import cv2
from scipy.io import loadmat


def get_dataset(dataset_name, shape, conversion, num_classes, batch_size, complex_dtype):
    def preprocess(image, label, conversion = conversion, shape = shape, num_classes = num_classes):
        x = image/255
        x = tf.image.resize(x, shape)
        mins = tf.math.reduce_min(x,axis=[1,2], keepdims=True)
        maxs = tf.math.reduce_max(x,axis=[1,2], keepdims=True)
        x = (x - mins)/(maxs-mins)
        x = x*conversion
        x = tf.ones(tf.shape(x), dtype=complex_dtype)*tf.math.exp(tf.cast(x, dtype=complex_dtype)*tf.constant(1j, dtype=complex_dtype))
        x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
        label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
        return x,label


    train_images = tfds.load(dataset_name, split='train', as_supervised=True, batch_size=batch_size)
    test_images = tfds.load(dataset_name, split='test', as_supervised=True, batch_size=batch_size)


    train_images = train_images.map(preprocess)
    test_images = test_images.map(preprocess)

    return train_images, test_images


def preprocess_measurements_lab(im_path, shape, float_dtype):
    if os.path.exists(im_path):
        image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE) #read image from im_path
        image = tf.constant(image, shape = (1, *image.shape, 1), dtype= float_dtype)/255
        return tf.image.resize(image, shape)
    else: 
        raise Exception(" Image path does not exists: - ", im_path)

def read_mask_used_in_lab(mask_path):
    if os.path.exists(mask_path):
        mascara = loadmat(mask_path)["phase_mask"]
        return mascara
    else: 
        raise Exception("Mask path does not exists: - ", mask_path)