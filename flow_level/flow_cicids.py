# -*- coding: utf-8 -*-
"""
@author: ssaka0
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Reshape, Flatten, Dense, Dropout, Bidirectional, LSTM, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import glob

# Load Dataset
cicids_files = glob.glob("CICIDS2017\\MachineLearningCVE\\" + "*.csv")
df= pd.concat([pd.read_csv(file) for file in cicids_files], ignore_index=True)

df[' Label'] = df[' Label'].apply(lambda x: 'WA-Brute Force' if x == 'Web Attack � Brute Force' else x)

df[' Label'] = df[' Label'].apply(lambda x: 'WA-XSS' if x == 'Web Attack � XSS' else x)

df[' Label'] = df[' Label'].apply(lambda x: 'WA-SqlInjection' if x == 'Web Attack � Sql Injection' else x)

df[' Label'].value_counts().plot(kind='bar')


x = df.drop([' Label'], axis=1)
y = df[[' Label']]

label_encoder = LabelEncoder()
y[' Label'] = label_encoder.fit_transform(y)

x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


desired_count_over = 5000
desired_count_under = 30000 
oversample_strategy = {i: desired_count_over for i in range(len(y.value_counts())) if y.value_counts()[i] < desired_count_over}
undersample_strategy = {i: desired_count_under for i in range(len(y.value_counts())) if y.value_counts()[i] > desired_count_under}

# Create the SMOTE and RandomUnderSampler objects
smote = SMOTE(sampling_strategy=oversample_strategy)
undersample = RandomUnderSampler(sampling_strategy=undersample_strategy)

# Combine SMOTE and RandomUnderSampler in a pipeline
pipeline = Pipeline(steps=[('smote', smote), ('undersample', undersample)])

# Print class distribution before resampling
print("Before resampling:", y.value_counts())
print()

x_resampled, y_resampled = pipeline.fit_resample(x_scaled, y)

# Print class distribution after resampling
print("After resampling:", y_resampled.value_counts())

x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)


y_train_one_hot = to_categorical(y_train, num_classes=15)
y_test_one_hot = to_categorical(y_test, num_classes=15)

model = Sequential()
# Input layer with BatchNormalization
model.add(Dense(128, activation='relu', input_shape=(78,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden layer 1
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden layer 2
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Hidden layer 3
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(15, activation='softmax'))  # Adjust units for your number of classes

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("Training DNN model...")
model.fit(x_train_scaled, y_train_one_hot, epochs=50, batch_size=256)

model.save('cicids_flow_dnn.keras')

y_pred = model.predict(x_test_scaled)

print("\nDNN Classification Report:")
print(classification_report(y_test_one_hot.argmax(axis=1), y_pred.argmax(axis=1)))

with open('classification_report_dnn.txt', 'w') as f:
    f.write(classification_report(y_test_one_hot.argmax(axis=1), y_pred.argmax(axis=1)))
    
    
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_one_hot.argmax(axis=1), y_pred.argmax(axis=1), normalize='true'), display_labels=label_encoder.classes_)

disp.plot(cmap='Blues', xticks_rotation=80, include_values=False)
disp.ax_.set_title("DNN Confusion Matrix")

X_train_cnn = np.expand_dims(x_train_scaled, axis=2)
X_test_cnn = np.expand_dims(x_test_scaled, axis=2)

cnn_model = Sequential()
cnn_model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(78, 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(32, kernel_size=3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(16, kernel_size=3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(15, activation='softmax'))

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training CNN model...")
cnn_model.fit(X_train_cnn, y_train_one_hot, epochs=50, batch_size=256)

model.save('cicids_flow_cnn.keras')


y_pred_cnn = cnn_model.predict(X_test_cnn)

print("\nCNN Classification Report:")
print(classification_report(y_test_one_hot.argmax(axis=1), y_pred_cnn.argmax(axis=1)))

with open('classification_report_cnn.txt', 'w') as f:
    f.write(classification_report(y_test_one_hot.argmax(axis=1), y_pred_cnn.argmax(axis=1)))
    
    
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_one_hot.argmax(axis=1), y_pred_cnn.argmax(axis=1), normalize='true'), display_labels=label_encoder.classes_)

disp.plot(cmap='Blues', xticks_rotation=80, include_values=False)
disp.ax_.set_title("CNN Confusion Matrix")


