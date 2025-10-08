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
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load Dataset
train = pd.read_csv("UNSW-NB15\\OneDrive_2024-02-16\\CSV Files\\Training and Testing Sets\\UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW-NB15\\OneDrive_2024-02-16\\CSV Files\\Training and Testing Sets\\UNSW_NB15_testing-set.csv")
data = pd.concat((train, test), axis=0)

data['attack_cat'].value_counts().plot(kind='bar')


# Encode categorical features
label_encoders = {}
for column in ['proto', 'service', 'state', 'attack_cat']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


x = data.drop(['attack_cat', 'label', 'id'], axis=1)
y = data[['attack_cat']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


desired_count_over = 30000
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


y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

model = Sequential()
# Input layer with BatchNormalization
model.add(Dense(128, activation='relu', input_shape=(42,)))
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

model.add(Dense(10, activation='softmax'))  # Adjust units for your number of classes

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("Training DNN model...")
model.fit(x_train_scaled, y_train_one_hot, epochs=100, batch_size=256)

model.save('unsw_flow_dnn.keras')

y_pred = model.predict(x_test_scaled)

print("\nDNN Classification Report:")
print(classification_report(y_test_one_hot.argmax(axis=1), y_pred.argmax(axis=1)))

with open('classification_report_dnn.txt', 'w') as f:
    f.write(classification_report(y_test_one_hot.argmax(axis=1), y_pred.argmax(axis=1)))
    
    
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_one_hot.argmax(axis=1), y_pred.argmax(axis=1), normalize='true'), display_labels=label_encoders['attack_cat'].classes_)

disp.plot(cmap='Blues', xticks_rotation=80, include_values=False)
disp.ax_.set_title("DNN Confusion Matrix")


X_train_cnn = np.expand_dims(x_train_scaled, axis=2)
X_test_cnn = np.expand_dims(x_test_scaled, axis=2)

cnn_model = Sequential()
cnn_model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(42, 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(32, kernel_size=3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(16, kernel_size=3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training CNN model...")
cnn_model.fit(X_train_cnn, y_train_one_hot, epochs=50, batch_size=256)

model.save('unsw_flow_cnn.keras')

y_pred_cnn = cnn_model.predict(X_test_cnn)

print("\nCNN Classification Report:")
print(classification_report(y_test_one_hot.argmax(axis=1), y_pred_cnn.argmax(axis=1)))

with open('classification_report_cnn.txt', 'w') as f:
    f.write(classification_report(y_test_one_hot.argmax(axis=1), y_pred_cnn.argmax(axis=1)))
    
    
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_one_hot.argmax(axis=1), y_pred_cnn.argmax(axis=1), normalize='true'), display_labels=label_encoders['attack_cat'].classes_)

disp.plot(cmap='Blues', xticks_rotation=80, include_values=False)
disp.ax_.set_title("CNN Confusion Matrix")


