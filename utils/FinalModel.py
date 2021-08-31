import tensorflow as tf 
from Muestreos import A_Fran, AT_Fran
from InitializationLayer import FSI_Initial,FSI_cell
from pruebas_layers import BackPropagationLayer
from AcquisitionLayer import Muestreo
from Generador import Unet
import numpy as np
import os

def normalizar(x):
  mins = tf.math.reduce_min(x, [1,2], keepdims=True)
  maxs = tf.math.reduce_max(x, [1,2], keepdims=True)
  return tf.math.divide(tf.math.subtract(x, mins), tf.math.subtract(maxs, mins))

class customGaussian(tf.keras.layers.Layer):
    def __init__(self, snr=10, name="Gaussian_noise_layer"):
        super(customGaussian, self).__init__(name=name)
        self.snr = snr
        
    def add_noise_each_x(self, x):

        m = x.shape[0]*x.shape[1]*x.shape[2]
        divisor = m*10**(self.snr/10)
        stddev = tf.math.sqrt(tf.math.divide(tf.math.pow(tf.norm(x, 'fro', axis= [0,1]),2), divisor))
        return x + tf.keras.backend.random_normal(shape=x.shape,mean=0,stddev=stddev, dtype=x.dtype)

    def call(self, input):
        input = tf.cast(input, tf.float64)
        
        salida = tf.map_fn(self.add_noise_each_x,input)
        
        return salida

class CLASSIFICATION_MODEL(tf.keras.Model):
    def __init__(self, shape, num_clases, L = 1, tipo_muestreo="ASM", sftf_folder=os.path.join("..", "SFTF_folder"), snr=10):
        super(CLASSIFICATION_MODEL, self).__init__()
        self.L = L
        self.muestreo_layer = Muestreo(self.L, tipo_muestreo=tipo_muestreo, sftf_folder=sftf_folder, trainable=False)
        self.classification_network = tf.keras.applications.MobileNetV2(input_shape=(*shape,3), classes=num_clases, weights=None, classifier_activation="softmax")
        # self.classification_network = tf.keras.applications.inception_v3.InceptionV3(input_shape=(*shape,3), classes=num_clases, weights=None, classifier_activation="softmax")
        self.conv_initial = tf.keras.layers.Conv2D(3, 3, padding="same",activation = None, name = "ConvInitial")
        self.ruido = customGaussian(snr = snr)
        #self.LRL1 = tf.keras.layers.LeakyReLU(alpha = 0.5, name= "LR_L")

    def build(self, input_shape):
        super(CLASSIFICATION_MODEL, self).build(input_shape)
        self.S = input_shape
        
    def call(self, input):
        input = tf.cast(input, tf.float32)
        muestras,_ = self.muestreo_layer(input)
        muestras = self.ruido(muestras)
        muestras = self.conv_initial(muestras)
        clasificion = self.classification_network(muestras)
        return clasificion

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class CLASSIFICATION_MODEL_with_back(tf.keras.Model):
    def __init__(self, shape, num_clases, L = 1, tipo_muestreo="ASM", sftf_folder=os.path.join("..", "SFTF_folder"),  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", snr=10):
        super(CLASSIFICATION_MODEL_with_back, self).__init__()
        self.L = L
        self.muestreo_layer = Muestreo(self.L, tipo_muestreo=tipo_muestreo, sftf_folder=sftf_folder,trainable=False)
        self.backpropagation_layer  = BackPropagationLayer(name = "BackPropagationLayer",sftf_folder=sftf_folder, tipo_muestreo = tipo_muestreo, archivo_sftf_solver = archivo_sftf_solver)
        self.classification_network = tf.keras.applications.MobileNetV2(input_shape=(*shape,3), classes=num_clases, weights=None, classifier_activation="softmax")
        # self.classification_network = tf.keras.applications.inception_v3.InceptionV3(input_shape=(*shape,3), classes=num_clases, weights=None, classifier_activation="softmax")
        self.conv_initial = tf.keras.layers.Conv2D(3, 3, padding="same",activation = None, name = "ConvInitial")
        self.ruido = customGaussian(snr = snr)
        


    def build(self, input_shape):
        super(CLASSIFICATION_MODEL_with_back, self).build(input_shape)
        self.S = input_shape
        

    def call(self, input):
        input = tf.cast(input, tf.float32)
        muestras, mask = self.muestreo_layer(input)
        muestras = self.ruido(muestras)
        back = self.backpropagation_layer([muestras, mask])
        real, imag = tf.unstack(back, num=2, axis=3)
        complex_back = tf.expand_dims(tf.complex(real, imag), -1)
        polar_back = tf.concat([tf.math.abs(complex_back), tf.math.angle(complex_back)], -1)

        muestras = tf.map_fn(normalizar, tf.transpose(muestras, perm=[3,0,1,2]))
        back = tf.map_fn(normalizar, tf.transpose(back, perm=[3,0,1,2]))
        polar_back = tf.map_fn(normalizar, tf.transpose(polar_back, perm=[3,0,1,2]))      
        caracteristicas = tf.concat([muestras,back,polar_back], 0)
        caracteristicas = tf.transpose(caracteristicas, perm=[1,2,3,0])
        caracteristicas = self.conv_initial(caracteristicas)
        clasificion = self.classification_network(caracteristicas)
        return clasificion

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class CLASSIFICATION_MODEL_with_initialization(tf.keras.Model):
    def __init__(self, shape, num_clases, L = 1, tipo_muestreo="ASM", sftf_folder=os.path.join("..", "SFTF_folder"), archivo_sftf = "SFTF_LAB_simul.mat",  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", snr=10):
        super(CLASSIFICATION_MODEL_with_initialization, self).__init__()
        self.L = L
        self.inicializacion = make_inicializacion(shape, L = L, archivo_sftf = archivo_sftf, archivo_sftf_solver = archivo_sftf_solver)
        self.muestreo_layer = Muestreo(self.L, tipo_muestreo=tipo_muestreo, sftf_folder=sftf_folder, trainable=False)
        self.classification_network = tf.keras.applications.MobileNetV2(input_shape=(*shape,3), classes=num_clases, weights=None, classifier_activation="softmax")
        # self.classification_network = tf.keras.applications.inception_v3.InceptionV3(input_shape=(*shape,3), classes=num_clases, weights=None, classifier_activation="softmax")
        self.conv_initial = tf.keras.layers.Conv2D(3, 3, padding="same",activation = None, name = "ConvInitial")
        self.ruido = customGaussian(snr = snr)        

    def build(self, input_shape):
        super(CLASSIFICATION_MODEL_with_initialization, self).build(input_shape)
        self.S = input_shape
        

    def call(self, input):
        input = tf.cast(input, tf.float32)
        muestras, mask = self.muestreo_layer(input)
        
        muestras = self.ruido(muestras)
        
        back = self.inicializacion([muestras, mask])
        real, imag = tf.unstack(back, num=2, axis=3)
        complex_back = tf.expand_dims(tf.complex(real, imag), -1)
        polar_back = tf.concat([tf.math.abs(complex_back), tf.math.angle(complex_back)], -1)

        muestras = tf.map_fn(normalizar, tf.transpose(muestras, perm=[3,0,1,2]))
        back = tf.map_fn(normalizar, tf.transpose(back, perm=[3,0,1,2]))
        polar_back = tf.map_fn(normalizar, tf.transpose(polar_back, perm=[3,0,1,2]))

        caracteristicas = tf.concat([muestras,back,polar_back], 0)
        caracteristicas = tf.transpose(caracteristicas, perm=[1,2,3,0])
        caracteristicas = self.conv_initial(caracteristicas)
        clasificion = self.classification_network(caracteristicas)
        return clasificion

    def model(self):
        x = tf.keras.Input(shape = self.S[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))



class make_inicializacion(tf.keras.layers.Layer):
    def __init__(self, shape, L = 1, tipo_muestreo="ASM", sftf_folder=os.path.join("..", "SFTF_folder"), archivo_sftf = "SFTF_LAB_simul.mat",  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", snr=10):
        super(make_inicializacion, self).__init__()
        self.L = L
        self.ruido = customGaussian(snr = snr)
        self.init_initialzation = FSI_Initial(name = "init_initialzation")
        self.Initialation = FSI_cell(name = "InitializationLayer",sftf_folder=sftf_folder, tipo_muestreo =tipo_muestreo, archivo_sftf = archivo_sftf, archivo_sftf_solver = archivo_sftf_solver)
        

    def build(self, input_shape):
        super(make_inicializacion, self).build(input_shape)
        self.S = input_shape
        

    def call(self, input):
        muestras, mask = input
        muestras = self.ruido(muestras)
        Ytr, Z = self.init_initialzation(muestras)
       
        for i in range(10):
          Z,mask = self.Initialation([Ytr, Z, mask])
          
        S = Z.shape
        normest = tf.sqrt(tf.multiply(tf.math.reduce_sum(Z, axis=[1,2,3],keepdims=True), tf.cast(S[1]*S[2]*S[3] , Z.dtype)))
        Z = tf.multiply(Z,normest)
        Z = tf.transpose(Z,perm=[0,2,3,1])
        #Z = tf.math.divide(Z, tf.cast(tf.math.reduce_max(tf.math.abs(Z), axis=[1,2,3]), Z.dtype))
        back_real = tf.math.real(Z);
        back_imag = tf.math.imag(Z); 

        back = tf.concat([back_real, back_imag], axis = -1)
        return  back


class ejemplo_backpropagation(tf.keras.Model):
    def __init__(self, shape, L = 1, tipo_muestreo="ASM", sftf_folder=os.path.join("..", "SFTF_folder"),  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", snr=10):
        super(ejemplo_backpropagation, self).__init__()
        self.muestreo_layer = Muestreo(L, tipo_muestreo=tipo_muestreo, sftf_folder=sftf_folder, trainable=False)
        self.backpropagation_layer  = BackPropagationLayer(name = "BackPropagationLayer",sftf_folder=sftf_folder, tipo_muestreo = tipo_muestreo,  archivo_sftf_solver = archivo_sftf_solver)
        self.ruido = customGaussian(snr = snr)
        

    def build(self, input_shape):
        super(ejemplo_backpropagation, self).build(input_shape)
        self.S = input_shape
        
    def call(self, input):
        input = tf.cast(input, tf.float32)
        muestras, mask = self.muestreo_layer(input)
        
        muestras = self.ruido(muestras)
        
        back = self.backpropagation_layer([muestras, mask])
        
        real, imag = tf.unstack(back, num=2, axis=3)
        complex_back = tf.expand_dims(tf.complex(real, imag), -1)
        polar_back = tf.concat([tf.math.abs(complex_back), tf.math.angle(complex_back)], -1)

        muestras = tf.map_fn(normalizar, tf.transpose(muestras, perm=[3,0,1,2]))
        back = tf.map_fn(normalizar, tf.transpose(back, perm=[3,0,1,2]))
        polar_back = tf.map_fn(normalizar, tf.transpose(polar_back, perm=[3,0,1,2]))

        caracteristicas = tf.concat([muestras,back,polar_back], 0)
        caracteristicas = tf.transpose(caracteristicas, perm=[1,2,3,0])
        return caracteristicas

class ejemplo_inicializacion(tf.keras.Model):
    def __init__(self, shape, L = 1, tipo_muestreo="ASM", sftf_folder=os.path.join("..", "SFTF_folder"), archivo_sftf = "SFTF_LAB_simul.mat",  archivo_sftf_solver = "SFTF_solver_LAB_simul.mat", snr=10):
        super(ejemplo_inicializacion, self).__init__()
        self.muestreo_layer = Muestreo(L, tipo_muestreo=tipo_muestreo, sftf_folder=sftf_folder, archivo_sftf = archivo_sftf, trainable=False)
        self.inicializacion = make_inicializacion(shape, L = L, archivo_sftf = archivo_sftf, archivo_sftf_solver = archivo_sftf_solver)
        self.ruido = customGaussian(snr = snr)
        

    def build(self, input_shape):
        super(ejemplo_inicializacion, self).build(input_shape)
        self.S = input_shape
        
    def call(self, input):
        input = tf.cast(input, tf.float32)
        
        muestras, mask = self.muestreo_layer(input)
        
        
        muestras = self.ruido(muestras)   
        
        back = self.inicializacion([muestras, mask])
        real, imag = tf.unstack(back, num=2, axis=3)
        complex_back = tf.expand_dims(tf.complex(real, imag), -1)
        polar_back = tf.concat([tf.math.abs(complex_back), tf.math.angle(complex_back)], -1)

        muestras = tf.map_fn(normalizar, tf.transpose(muestras, perm=[3,0,1,2]))
        back = tf.map_fn(normalizar, tf.transpose(back, perm=[3,0,1,2]))
        polar_back = tf.map_fn(normalizar, tf.transpose(polar_back, perm=[3,0,1,2]))

        caracteristicas = tf.concat([muestras,back,polar_back], 0)
        caracteristicas = tf.transpose(caracteristicas, perm=[1,2,3,0])
        return caracteristicas