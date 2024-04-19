#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,f1_score, roc_curve, auc,roc_auc_score
from Include.MLSMOTE import get_tail_label,get_index,get_minority_instace,MLSMOTE
from scipy.stats import yeojohnson

#---------------------------------------------------------------------------------#
# --------------------------------Preprocessing DATA------------------------------#
#---------------------------------------------------------------------------------#

# Loading data
x_train = np.load('../Training data/X_train.npy') 
y_train = np.load('../Training data/y_train.npy')  
x_test=np.load("../Testing data/X_test.npy")

x_train = pd.DataFrame(x_train).rename(columns={i: f"x_{i}" for i in range(111)})
y_train = pd.DataFrame(y_train).rename(columns={i: f"y_{i}" for i in range(11)})
x_test = pd.DataFrame(x_test).rename(columns={i: f"x_{i}" for i in range(111)})

# Filling missing values with means
for column in x_train.columns:
    x_train.fillna({column: x_train[column].median()}, inplace=True)
    x_test.fillna({column: x_test[column].median()}, inplace=True)

# Filter out features that tilt largely
threshold = 0.01 
cols_to_drop = [col for col in x_train.columns if (x_train[col] != 0).mean() < threshold]

x_train.drop(columns=cols_to_drop, inplace=True)
x_test.drop(columns=cols_to_drop, inplace=True)

# Yeo-Johnson transformation
x_train = x_train.apply(lambda x: yeojohnson(x)[0])
x_test = x_test.apply(lambda x: yeojohnson(x)[0])

# Standardize the input data
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)


# Apply MLSMOTE
tail_labels = get_tail_label(y_train)
indices = get_index(y_train)
X_sub, y_sub = get_minority_instace(x_train, y_train)

X_resampled, y_resampled = MLSMOTE(X_sub, y_sub, 1)  # Generate new samples
X_train_final = pd.concat([x_train, X_resampled], ignore_index=True)
y_train_final = pd.concat([y_train, y_resampled], ignore_index=True)

# Split out train set and test set from original Train data
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_final, y_train_final, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------#
# -------------------------Configure MTL Model & Training-------------------------#
#---------------------------------------------------------------------------------#
# Define hard shared parameter layers 
shared_input = Input(shape=(X_train.shape[1], ), name='shared_input')
shared_layer = Dense(128, activation='relu')(shared_input)
shared_layer = Dense(128, activation='relu')(shared_layer)
shared_layer = Dense(64, activation='relu')(shared_layer)
shared_layer = Dense(32, activation='tanh')(shared_layer)

# Task specific layers
outputs = []
for i in range(11):
    task_output = Dense(32,  activation='tanh', name=f'task_{i}_hidden')(shared_layer)
    task_output = Dense(1,  activation='sigmoid', name=f'task_{i}_output')(task_output)
    outputs.append(task_output)
outputs = Concatenate(axis=-1)(outputs)

# Configure model and compile arguments
model = Model(inputs=shared_input, outputs=outputs) 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ])

# Training
history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_valid, Y_valid))


#---------------------------------------------------------------------------------#
# -------------------------Calculate Loss and Evaluate Model----------------------#
#---------------------------------------------------------------------------------#
train_precision = history.history['precision'];     val_precision = history.history['val_precision']
train_recall = history.history['recall'];           val_recall = history.history['val_recall']
train_loss = history.history['loss'];               val_loss = history.history['val_loss']

Y_pred = model.predict(X_valid)
Y_pred_binary = (Y_pred > 0.5).astype(int)

macro_f1 = f1_score(Y_valid, Y_pred_binary, average='macro')
micro_f1 = f1_score(Y_valid, Y_pred_binary, average='micro')
print("Macro F1 Score:", macro_f1)
print("Micro F1 Score:", micro_f1)

print(f'\nTrain Precision: {train_precision[-1]}')
print(f'Validation Recall: {val_recall[-1]}\n')

print(f'Train Recall: {train_recall[-1]}')
print(f'Validation Precision: {val_precision[-1]}\n')

print(f'Train Loss: {train_loss[-1]}')
print(f'Test Loss: {val_loss[-1]}')

#---------------------------------------------------------------------------------#
# -----------------------------Predcit test data and show-------------------------#
#---------------------------------------------------------------------------------#
y_hat = pd.DataFrame(model.predict(x_test)).rename(columns={i: f"y_pred_{i}" for i in range(11)})

print(y_hat)