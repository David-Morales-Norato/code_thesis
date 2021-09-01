'''
import the necessary packages
'''
from utils.dataio import get_dataset
from model.AcquisitionLayer import Muestreo
from model.InitializationLayer import BackPropagationLayer, FSI_Initial, FSI_cell
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import sys
import os

def plot_image(image, path2save):
    plt.imshow(image)
    plt.colorbar()
    plt.savefig(path2save)
    plt.close()

''' 
parameters
'''

config_file = sys.argv[-1]
if os.path.exists(config_file):
    print("opening config file")
    with open(config_file) as json_file:
        params = json.load(json_file)
        data_params, forward_params, training_params, other_params = params["data"], params["forward_model"], params["training"], params["other"]
else:
    raise Exception("Config file not found: "+ config_file)

# data
SHAPE = data_params["image_shape"]
MASK_PATH = data_params["mask_path"]
DATASET_NAME = data_params["dataset_name"]
NUM_CLASES = data_params["num_classes"]


# Forward model parameters
SNAPSHOTS =  forward_params["number_snapshots"]
TIPO_MUESTREO =  forward_params["tipo_muestreo"]
KAPPA =  forward_params["kappa"]
WAVE_LENGTH =  forward_params["wavelength"]
DX =  forward_params["pixel_size"]
DISTANCE_SENSOR =  forward_params["distance_sensor"]
NORMALIZE_MEASUREMENTS = forward_params["normalize_measurements"]

# Training 
BATCH_SIZE = training_params["batch_size"]
RESULTS_FOLDER =  os.path.join(training_params["results_folder"], DATASET_NAME)
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# other
FLOAT_DTYPE =  "float32"
COMPLEX_DTYPE =  "complex64"
P =  other_params["p"]
K_SIZE = other_params["k_size"]
N_ITER = other_params["n_iterations"]
CONVERSION = other_params["conversion"]*np.pi


''' 
Data preprocessing
'''
if (DATASET_NAME == "mnist" or DATASET_NAME == "fashion_mnist"):
    _, dataset = get_dataset(dataset_name = DATASET_NAME, shape = SHAPE, conversion = CONVERSION, num_classes = NUM_CLASES, batch_size = BATCH_SIZE, complex_dtype=COMPLEX_DTYPE)
    mascara_lab = None
    x_image, _ = next(iter(dataset))
    #x_image = x_image[0]
elif  (DATASET_NAME == "lab_measurements"):
    x_image = tf.zeros((1, *SHAPE,2), dtype=FLOAT_DTYPE)
    muestras_entrada_lab = preprocess_measurements_lab(im_path = IM_PATH, shape = SHAPE, float_dtype = FLOAT_DTYPE)
    mascara_lab = read_mask_used_in_lab(mask_path = MASK_PATH)
else:
    raise Exception("invalid type image")


''' 
Acquisition Layer definition
'''
muestreo_layer = Muestreo(snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
                dx = DX,distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
                kappa = KAPPA, mascara_lab = mascara_lab, normalize_measurements = NORMALIZE_MEASUREMENTS, trainable_mask=False, 
                float_dtype = FLOAT_DTYPE, complex_dtype=COMPLEX_DTYPE, name="Muestreo")


muestras_entrada, mascara_usada, var_de_interes = muestreo_layer(x_image)

if  (DATASET_NAME == "only_phase_from_lab"):
    muestras_entrada = muestras_entrada_lab


'''
image and matrices
'''
x_complex = tf.complex(x_image[0,...,0], x_image[0,...,1])

plot_image(x_image[0,...,0], os.path.join(RESULTS_FOLDER, "imagen_real.png"))
plot_image(x_image[0,...,1], os.path.join(RESULTS_FOLDER, "imagen_imag.png"))
plot_image(tf.math.abs(x_complex), os.path.join(RESULTS_FOLDER, "imagen_abs.png"))
plot_image(tf.math.angle(x_complex), os.path.join(RESULTS_FOLDER, "imagen_angle.png"))

if (TIPO_MUESTREO=="ASM"):
    plot_image(tf.math.abs(tf.squeeze(var_de_interes)), os.path.join(RESULTS_FOLDER, "sftf_abs.png"))
    plot_image(tf.math.angle(tf.squeeze(var_de_interes)), os.path.join(RESULTS_FOLDER, "sftf_angle.png"))
elif (TIPO_MUESTREO=="FRESNELL"):
    plot_image(tf.math.abs(tf.squeeze(var_de_interes[0])), os.path.join(RESULTS_FOLDER, "Q1_abs.png"))
    plot_image(tf.math.angle(tf.squeeze(var_de_interes[0])), os.path.join(RESULTS_FOLDER, "Q1_angle.png"))
    plot_image(tf.math.abs(tf.squeeze(var_de_interes[1])), os.path.join(RESULTS_FOLDER, "Q2_abs.png"))
    plot_image(tf.math.angle(tf.squeeze(var_de_interes[1])), os.path.join(RESULTS_FOLDER, "Q2_angle.png"))
plot_image(tf.math.abs(tf.squeeze(mascara_usada)), os.path.join(RESULTS_FOLDER, "mascara_abs.png"))
plot_image(tf.math.angle(tf.squeeze(mascara_usada)), os.path.join(RESULTS_FOLDER, "mascara_angle.png"))
plot_image(tf.squeeze(muestras_entrada), os.path.join(RESULTS_FOLDER, "muestras_imagen.png"))


'''
run back proopagation
'''

backpropagation_layer  = BackPropagationLayer(tipo_muestreo = TIPO_MUESTREO, complex_dtype = COMPLEX_DTYPE, name = "BackPropagationLayer")
back = backpropagation_layer([muestras_entrada,mascara_usada,var_de_interes])
back_real, back_imag = tf.unstack(back, num=2, axis=-1)
back_complex = tf.complex(back_real[0,], back_imag[0,])

plot_image(tf.squeeze(back_real), os.path.join(RESULTS_FOLDER, "back_real.png"))
plot_image(tf.squeeze(back_imag), os.path.join(RESULTS_FOLDER, "back_imag.png"))
plot_image(tf.math.abs(back_complex), os.path.join(RESULTS_FOLDER, "back_abs.png"))
plot_image(tf.math.angle(back_complex), os.path.join(RESULTS_FOLDER, "back_angle.png"))


'''
run fsi initialization
'''
init_initialzation = FSI_Initial(p = P, float_dtype=FLOAT_DTYPE,complex_dtype=COMPLEX_DTYPE, name = "init_initialzation")
Initialation = FSI_cell(p = P, k_size = K_SIZE, kappa=KAPPA, tipo_muestreo = TIPO_MUESTREO,float_dtype=FLOAT_DTYPE,complex_dtype=COMPLEX_DTYPE, name = "initialization_cell")
Ytr, Z = init_initialzation(muestras_entrada)
for i in range(10):
  Z = Initialation([Ytr, Z, mascara_usada, var_de_interes])

Z_abs = tf.squeeze(tf.math.abs(Z))
Z_angle = tf.squeeze(tf.math.angle(Z))

plot_image(tf.squeeze(tf.math.real(Z)), os.path.join(RESULTS_FOLDER, "init_real.png"))
plot_image(tf.squeeze(tf.math.imag(Z)), os.path.join(RESULTS_FOLDER, "init_imag.png"))
plot_image(tf.math.abs(tf.squeeze(Z)), os.path.join(RESULTS_FOLDER, "init_abs.png"))
plot_image(tf.math.angle(tf.squeeze(Z)), os.path.join(RESULTS_FOLDER, "init_angle.png"))


print("The code ran successfully")

