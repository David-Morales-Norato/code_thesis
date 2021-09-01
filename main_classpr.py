'''
import the necessary packages
'''

from utils.dataio import get_dataset
from model.FinalModel import CLAS_PR, CLAS_PR_BACK, CLAS_PR_INIT
from utils.custom_callbacks import callback_class
import tensorflow as tf
import numpy as np
import json
import sys
import os

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
    print("Config file", config_file)
    print("actual path", os.getcwd())
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
SNR = forward_params["snr"]

# Training 
MODEL_TYPE = training_params["model_type"]
BATCH_SIZE =  training_params["batch_size"]
EPOCHS =  training_params["number_epochs"]
N_SAVES =  training_params["number_saves"]
LR =  training_params["learning_rate"]
# LOSS_1_WEIGHT =  training_params["loss_1_weight"]
# LOSS_2_WEIGHT =  training_params["loss_2_weight"]
RESULTS_FOLDER =  os.path.join(training_params["results_folder"], DATASET_NAME, MODEL_TYPE)
WEIGHTS_PATH =  os.path.join(RESULTS_FOLDER, training_params["weights_path"])

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
    train_dataset, test_dataset = get_dataset(dataset_name = DATASET_NAME, shape = SHAPE, conversion = CONVERSION, num_classes = NUM_CLASES, batch_size = BATCH_SIZE, complex_dtype=COMPLEX_DTYPE)
    train_dataset = train_dataset.take(64)
    test_dataset = test_dataset.take(64)
    mascara_lab = None
else:
    raise Exception("invalid type dataset")


'''
Model definition
'''
if MODEL_TYPE == "None":
    print(" ---------------- using model without any initialization ---------------- ")
    modelo_class = CLAS_PR(shape = SHAPE, num_classes = NUM_CLASES, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
            dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, kappa = KAPPA, mascara_lab = mascara_lab, 
            normalize_measurements = NORMALIZE_MEASUREMENTS, trainable_mask = False, 
            float_dtype = FLOAT_DTYPE, complex_dtype = COMPLEX_DTYPE, snr = SNR)
    modelo_class.build((BATCH_SIZE, *SHAPE, 2))
elif MODEL_TYPE == "back_propagation":
    print(" ---------------- using model using the back propagation operator ---------------- ")
    modelo_class = CLAS_PR_BACK(shape = SHAPE, num_classes = NUM_CLASES, snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, 
            dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, kappa = KAPPA, mascara_lab = mascara_lab, 
            normalize_measurements = NORMALIZE_MEASUREMENTS, trainable_mask = False, 
            float_dtype = FLOAT_DTYPE, complex_dtype = COMPLEX_DTYPE, snr = SNR)
    modelo_class.build((BATCH_SIZE, *SHAPE, 2))
elif MODEL_TYPE == "fsi_initialization":
    print(" ---------------- using model using the FSI initialization algorithm---------------- ")
    modelo_class = CLAS_PR_INIT(shape = SHAPE, num_classes = NUM_CLASES, p = P, k_size = K_SIZE, n_iterations = N_ITER, 
            snapshots = SNAPSHOTS, wave_length = WAVE_LENGTH, dx = DX, distance = DISTANCE_SENSOR, tipo_muestreo = TIPO_MUESTREO, 
            kappa = KAPPA, mascara_lab = mascara_lab, normalize_measurements = NORMALIZE_MEASUREMENTS, trainable_mask = False, 
            float_dtype = FLOAT_DTYPE, complex_dtype = COMPLEX_DTYPE, snr = SNR)
    modelo_class.build((BATCH_SIZE, *SHAPE, 2))
else:
    raise Exception("invalid model type")

print(modelo_class.summary())

'''
metrics and callbacks
'''
custom_callback = callback_class(RESULTS_FOLDER, generator = test_dataset, ext='.png', num_epochs = EPOCHS, num_saves = N_SAVES)
chekpoint = tf.keras.callbacks.ModelCheckpoint(WEIGHTS_PATH, save_weights_only = True)

'''
training loop
'''
opti = tf.keras.optimizers.Adam(amsgrad = True, learning_rate = LR)
modelo_class.compile(optimizer = opti, loss = "binary_crossentropy", metrics = [tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"), tf.keras.metrics.Recall(name = "recall_1"), tf.keras.metrics.Precision(name = "precision_1")])
history = modelo_class.fit(x = train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks = [chekpoint, custom_callback], verbose=2)