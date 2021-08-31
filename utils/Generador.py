import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, n_filters=32, k_size=3, alpha = 0.5, name="Encoder", bloque = 1, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.n_filters = n_filters
        self.k_size = k_size
        self.alpha = alpha
        self.bn = tf.keras.layers.BatchNormalization(name = "DBN_"+str(bloque))
        self.conv_1 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "DConv"+"_B"+str(bloque)+"_1")
        self.LRL1 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "DLRL"+"_B"+str(bloque)+"_1")
        self.conv_2 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "DConv"+"_B"+str(bloque)+"_2")
        self.LRL2 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "DLRL"+"_B"+str(bloque)+"_2")
        self.bn2 = tf.keras.layers.BatchNormalization(name = "DBN_"+str(bloque)+"_2")
        #self.conv_3 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "DConv"+"_B"+str(bloque)+"_3")
        #self.LRL3 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "DLRL"+"_B"+str(bloque)+"_3")
        self.MP = tf.keras.layers.MaxPool2D((2, 2), (2, 2), name="MP_B_"+str(bloque))

    def call(self, inputs):

        c = self.conv_1(inputs)
        c = self.LRL1(c)
        c = self.bn(c)
        c = self.conv_2(c)
        c = self.LRL2(c)
        skip = self.bn2(c)
        salida = self.MP(skip)
        return [skip, salida]

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, n_filters=32, k_size=3, alpha = 0.2, name="Decoder", bloque = 1,**kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.n_filters = n_filters
        self.k_size = k_size
        self.alpha = alpha
        self.bn = tf.keras.layers.BatchNormalization(name = "UBN_"+str(bloque))
        self.conv_t = tf.keras.layers.Conv2DTranspose(self.n_filters, self.k_size, padding="same", strides = (2, 2), name= "UTConv"+"_B"+str(bloque)+"_1")
        self.LRL1 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "ULRL"+"_B"+str(bloque)+"_1")
        self.conv_2 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "UConv"+"_B"+str(bloque)+"_2")
        self.LRL2 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "ULRL"+"_B"+str(bloque)+"_2")
        self.bn2 = tf.keras.layers.BatchNormalization(name = "DBN_"+str(bloque)+"_2")
        # self.conv_3 = tf.keras.layers.Conv2D(self.n_filters, self.k_size, padding="same", name= "UConv"+"_B"+str(bloque)+"_3")
        # self.LRL3 = tf.keras.layers.LeakyReLU(alpha = self.alpha, name= "ULRL"+"_B"+str(bloque)+"_3")
        self.concat = tf.keras.layers.Concatenate(name="Concat_B_"+str(bloque))

    def call(self, inputs):

        skip, entrada = inputs
        c = self.conv_t(entrada)
        up_sampled = self.LRL1(c)
        concat = self.concat([up_sampled, skip])
        c = self.bn(concat)
        c = self.conv_2(c)
        c = self.LRL2(c)
        salida = self.bn2(c)
        return salida

class Unet(tf.keras.layers.Layer):
    
    def __init__(self, k_size=3, alpha = 0.5, filtros_encoder = [64, 128,1024], filtros_decoder = [1024, 128, 64] , name="UNET", **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)
        
        self.k_size = k_size
        self.alpha = alpha

        self.filtros_encoder = filtros_encoder
        self.filtros_decoder = filtros_decoder

        # Bottleneck
        self.conv_1 = tf.keras.layers.Conv2D(self.filtros_encoder[-1], 1, padding="same", name = "Conv_BottleNEck_1")
        self.LRL1 = tf.keras.layers.LeakyReLU(alpha=0.2, name = "B_LRL_1")
        self.conv_2 = tf.keras.layers.Conv2D(self.filtros_encoder[-1], 1, padding="same", name = "Conv_BottleNEck_2")
        self.LRL2 = tf.keras.layers.LeakyReLU(alpha=0.2, name = "B_LRL_2")


        self.encoder_1 = Encoder(self.filtros_encoder[0], name = "Encoder_1", bloque=1)
        self.encoder_2 = Encoder(self.filtros_encoder[1], name = "Encoder_2", bloque=1)
        self.encoder_3 = Encoder(self.filtros_encoder[2], name = "Encoder_3", bloque=1)

        self.dencoder_1 = Decoder(self.filtros_decoder[0], name = "Decoder_1", bloque=1)
        self.dencoder_2 = Decoder(self.filtros_decoder[1], name = "Decoder_2", bloque=1)
        self.dencoder_3 = Decoder(self.filtros_decoder[2], name = "Decoder_3", bloque=1)

                          
    def call(self, inputs):
        
        s1, x = self.encoder_1(inputs)
        s2, x = self.encoder_2(x)
        s3, x = self.encoder_3(x)
        # s3,x = self.encoder_4(x)
        # s4,x = self.encoder_5(x)
 
        b = self.conv_1(x)
        b = self.LRL1(b)
        b = self.conv_2(b)
        b = self.LRL2(b)

        c = self.dencoder_1([s3, b])
        c = self.dencoder_2([s2, c])
        c = self.dencoder_3([s1, c])
        # c = self.dencoder_4([s1, c])
        # salida_unet = self.dencoder_5([s0, c])

        salida_unet = c
        return salida_unet

    def model(self, input_shape):
        x = tf.keras.Input(shape = input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

