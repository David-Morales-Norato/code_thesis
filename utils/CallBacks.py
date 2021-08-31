import json
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


class callback_test_clasification(tf.keras.callbacks.Callback):

  def __init__(self, results_folder, num_epochs, generator, ext='.png'):
    if (not os.path.exists(results_folder)):
      print("Creando directorio de resultados en : ", results_folder)
      os.makedirs(results_folder)
    self.generator = generator

    self.loss_hist =  os.path.join(results_folder, "loss_hist"+ ext)
    self.acc_hist =  os.path.join(results_folder, "acc_hist"+ ext)
    self.f1_hist =  os.path.join(results_folder, "f1_hist"+ ext)
    self.precision_hist =  os.path.join(results_folder, "precision_hist"+ ext)
    self.recall_hist =  os.path.join(results_folder, "recall_hist"+ ext)
    self.output_csv_file = os.path.join(results_folder, "data.csv")

    self.train_loss_hist = []
    self.val_loss_hist = []
    self.test_loss_hist = []

    self.train_acc_hist = []
    self.val_acc_hist = []
    self.test_acc_hist = []

    self.train_f1_hist = []
    self.val_f1_hist = []
    self.test_f1_hist = []
    
    self.train_precision_hist = []
    self.val_precision_hist = []
    self.test_precision_hist = []

    self.train_recall_hist = []
    self.val_recall_hist = []
    self.test_recall_hist = []

    self.acc_m = tf.keras.metrics.BinaryAccuracy()
    self.recall_m = tf.keras.metrics.Recall()
    self.precision_m = tf.keras.metrics.Precision()
    self.columns = ["epoch", "loss","val_loss","binary_accuracy","val_binary_accuracy","test_binary_accuracy","precision_1","val_precision_1","test_precision_1","recall_1","val_recall_1","test_recall_1","f1_score","val_f1_score","test_f1_score"]
    self.data_frame = pd.DataFrame(columns = self.columns,
                                  index = np.arange(num_epochs))
                                    

  def save_plot(self, X, labels, name, epoch=-1):
    for l,n in zip(X,labels):
      plt.plot(l, label = n)
    plt.legend()
    plt.title('Epoch: ' + str(epoch))
    plt.savefig(name)
    plt.close()

  def f1_m(self,precision, recall):
    return 2*((precision*recall)/(precision+recall+1e-8))
   
  # def plot_confusion_matrix():
  #   pred_labels = np.array([])
  #   real_labels = np.array([])
  #   for image, label in test_images:
  #     label_pred = tf.argmax(modelo.predict(image),axis=1).numpy()
  #     label = tf.argmax(label,axis=1).numpy()
  #     pred_labels = np.concatenate((label_pred,pred_labels),axis=0)
  #     real_labels = np.concatenate((label,real_labels),axis=0)

  #   cf_matrix = tf.math.confusion_matrix(real_labels, pred_labels, num_classes=100)
  #   cf_matrix = cf_matrix/np.sum(cf_matrix, axis=1)[:, np.newaxis]
  #   plt.figure(figsize=(12,9))
  #   sns.heatmap(cf_matrix, annot=True, vmin=0)
  #   plt.savefig(os.path.join(results_folder, "confusion_matrix.svg"))

  def on_epoch_end(self, epoch, logs=None):
    

    X_f, y_true = next(iter(self.generator))
    y_pred = self.model(X_f)


    loss, val_loss = logs['loss'],logs['val_loss']
    binary_accuracy, val_binary_accuracy, test_acc = logs['binary_accuracy'],logs['val_binary_accuracy'], self.acc_m(y_true, y_pred).numpy()
    precision_1, val_precision_1, test_precision = logs['precision_1'],logs['val_precision_1'], self.precision_m(y_true, y_pred).numpy()
    recall_1, val_recall_1, test_recall = logs['recall_1'],logs['val_recall_1'], self.recall_m(y_true, y_pred).numpy()
    f1_score, val_f1_score, test_f1_score = self.f1_m(logs['precision_1'], logs['recall_1']),self.f1_m(logs['val_precision_1'], logs['val_recall_1']), self.f1_m(test_precision,test_recall)
    #print(epoch, loss,val_loss,binary_accuracy,val_binary_accuracy,test_acc,precision_1,val_precision_1,test_precision,recall_1,val_recall_1,test_recall,f1_score,val_f1_score,test_f1_score)
    self.data_frame.iloc[epoch,:] = [epoch, loss,val_loss,binary_accuracy,val_binary_accuracy,test_acc,precision_1,val_precision_1,test_precision,recall_1,val_recall_1,test_recall,f1_score,val_f1_score,test_f1_score]
    self.data_frame.to_csv(self.output_csv_file,index=False,header=True)


    ax = self.data_frame.plot(y = ["loss","val_loss"]); ax.figure.savefig(self.loss_hist); fig = ax.get_figure(); plt.close(fig)
    ax = self.data_frame.plot(y = ["binary_accuracy","val_binary_accuracy","test_binary_accuracy"]); ax.figure.savefig(self.acc_hist); fig = ax.get_figure(); plt.close(fig)
    ax = self.data_frame.plot(y = ["precision_1","val_precision_1","test_precision_1"]); ax.figure.savefig(self.precision_hist); fig = ax.get_figure(); plt.close(fig)
    ax = self.data_frame.plot(y = ["recall_1","val_recall_1","test_recall_1"]); ax.figure.savefig(self.recall_hist); fig = ax.get_figure(); plt.close(fig)
    ax = self.data_frame.plot(y = ["f1_score","val_f1_score","test_f1_score"]); ax.figure.savefig(self.f1_hist); fig = ax.get_figure(); plt.close(fig)


