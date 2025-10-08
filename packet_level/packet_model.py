# -*- coding: utf-8 -*-
"""
Refactored Packet Classification Model
Focus on NN, CNN, and Proposed Model with detailed metrics per class
Created on October 8, 2025
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Add, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import time
import joblib
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import psutil
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def count(payload):
    """Convert payload to byte frequency vector"""
    vector = np.zeros(256, dtype=int)
    payload_list = list(str(payload).split(',')[:-1])
    count = Counter(payload_list)
    
    for num_str, cnt in count.items():
        try:
            num = int(num_str)
            if 0 <= num <= 255:
                vector[num] = cnt
        except ValueError:
            continue
    return vector


def profile_training(func, X_train, y_train):
    """Profile training time and memory usage"""
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_usage_before = process.memory_info().rss / (1024 ** 2)
   
    model = func(X_train, y_train)
    
    mem_usage_after = process.memory_info().rss / (1024 ** 2)
    end_time = time.time()

    time_taken = end_time - start_time
    memory_used = mem_usage_after - mem_usage_before
    
    return model, time_taken, memory_used


def train_neural_network(X_train, y_train, num_classes):
    """Train a simple Neural Network for multi-class classification"""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07), 
                 loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, to_categorical(y_train), epochs=100, batch_size=256, verbose=1)
    return model


def train_cnn(X_train, y_train, num_classes):
    """Train a CNN model for multi-class classification"""
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    
    cnn_model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    
    cnn_model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    
    cnn_model.add(Flatten())
    cnn_model.add(Dense(16, activation='relu'))
    cnn_model.add(Dense(num_classes, activation='softmax'))

    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
                     loss='categorical_crossentropy', metrics=['accuracy'])
    
    cnn_model.fit(X_train, to_categorical(y_train), epochs=30, batch_size=256, verbose=1)
    return cnn_model


def train_proposed_model(X_train, y_train, num_classes):
    """Train the proposed dual-path model with residual connections"""
    # Input layers
    input_1 = Input(shape=(5,))  # First 5 flow-level features
    input_2 = Input(shape=(12,))  # 12-dimensional abstract payload vector
    
    # High-Level Path (first 5 features)
    high_level = Dense(32, activation='relu')(input_1)
    high_level = BatchNormalization()(high_level)
    high_level = Dropout(0.2)(high_level)
    high_level = Dense(16, activation='relu')(high_level)
    
    # Low-Level Path (abstract payload vector)
    low_level = Dense(64, activation='relu')(input_2)
    low_level = BatchNormalization()(low_level)
    low_level = Dropout(0.2)(low_level)
    low_level = Dense(32, activation='relu')(low_level)
    
    # Merge the two paths
    merged = Concatenate()([high_level, low_level])
    
    # Enhanced Residual block with dense layers
    res_1 = Dense(64, activation='relu')(merged)
    res_1 = BatchNormalization()(res_1)
    res_2 = Dense(64, activation='relu')(res_1)
    res_2 = Dropout(0.2)(res_2)
    res_output = Add()([res_1, res_2])  # Residual connection
    
    # Fully connected layers after merge
    dense_3 = Dense(64, activation='relu')(res_output)
    dense_3 = Dropout(0.3)(dense_3)
    dense_4 = Dense(32, activation='relu')(dense_3)
    
    # Output layer (multi-class classification)
    output = Dense(num_classes, activation='softmax')(dense_4)
    
    # Model definition
    model = Model(inputs=[input_1, input_2], outputs=output)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-05),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    model.fit([X_train.iloc[:,:5], X_train.iloc[:,5:]], to_categorical(y_train), 
              epochs=100, batch_size=256, verbose=1)
    return model


def evaluate_model(model, X_test, y_test, model_name, label_encoder, is_proposed=False):
    """Evaluate model and print detailed metrics for each class"""
    print(f"\n{'='*20} {model_name} Results {'='*20}")
    
    # Prediction
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_usage_before = process.memory_info().rss / (1024 ** 2)
    
    if is_proposed:
        y_pred = model.predict([X_test.iloc[:,:5], X_test.iloc[:,5:]])
    else:
        if model_name == "CNN":
            X_test_reshaped = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
            y_pred = model.predict(X_test_reshaped)
        else:
            y_pred = model.predict(X_test)
    
    mem_usage_after = process.memory_info().rss / (1024 ** 2)
    end_time = time.time()
    
    time_taken = end_time - start_time
    memory_used = mem_usage_after - mem_usage_before
    
    # Convert predictions to class labels
    if len(y_pred.shape) == 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, average='macro', zero_division=0)
    
    print(f"Overall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"Prediction Time: {time_taken:.4f} seconds")
    print(f"Memory Used: {memory_used:.4f} MiB")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_test, y_pred_classes, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred_classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred_classes, average=None, zero_division=0)
    
    # Calculate per-class accuracy (diagonal of normalized confusion matrix)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    for i, class_name in enumerate(label_encoder.classes_):
        if i < len(precision_per_class):
            acc = per_class_accuracy[i] if i < len(per_class_accuracy) else 0.0
            prec = precision_per_class[i]
            rec = recall_per_class[i]
            f1_score_class = f1_per_class[i]
            
            print(f"{class_name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1_score_class:<10.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, zero_division=0))
    
    return accuracy, precision, recall, f1


def main():
    """Main function to train and evaluate models"""
    
    # Data paths - you can switch between CICIDS and UNSW datasets
    # CICIDS dataset
    train_path = "merged_headerand12Payload_cicids_train.csv"
    test_path = "cicids_merged_test.csv"
    pca_path = "cicids_pca_model.pkl"
    model_save_prefix = "cicids"
    
    # # UNSW dataset (uncomment to use)
    # train_path = "merged_headerand12Payload_unsw_train.csv"
    # test_path = "unsw_merged_test.csv"
    # pca_path = "unsw_pca_model.pkl"
    # model_save_prefix = "unsw"
    
    print("Loading and preprocessing training data...")
    train = pd.read_csv(train_path)
    
    # Prepare labels
    y_train = train['label']
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    
    # Prepare features
    train = train.drop('label', axis=1)
    protocol_encoder = LabelEncoder()
    train['protocol'] = protocol_encoder.fit_transform(train['protocol'])
    
    # Data balancing
    y_train = pd.Series(y_train)
    desired_count_over = 30000
    desired_count_under = 30000 
    oversample_strategy = {i: desired_count_over for i in range(len(y_train.value_counts())) 
                          if y_train.value_counts()[i] < desired_count_over}
    undersample_strategy = {i: desired_count_under for i in range(len(y_train.value_counts())) 
                           if y_train.value_counts()[i] > desired_count_under}

    smote = SMOTE(sampling_strategy=oversample_strategy)
    undersample = RandomUnderSampler(sampling_strategy=undersample_strategy)
    pipeline = Pipeline(steps=[('smote', smote), ('undersample', undersample)])

    print("Before resampling:", y_train.value_counts())
    x_resampled, y_resampled = pipeline.fit_resample(train, y_train)
    print("After resampling:", y_resampled.value_counts())
    
    # Models to train
    models_to_train = {
        "Neural_Network": lambda X_train, y_train: train_neural_network(X_train, y_train, num_classes=len(set(y_train))),
        "CNN": lambda X_train, y_train: train_cnn(X_train, y_train, num_classes=len(set(y_train))),
        "Proposed_Model": lambda X_train, y_train: train_proposed_model(X_train, y_train, num_classes=len(set(y_train)))
    }
    
    trained_models = {}
    
    # Train all models
    for name, train_func in models_to_train.items():
        print(f"\nTraining {name}...")
        model, time_taken, memory_used = profile_training(train_func, x_resampled, y_resampled)
        
        print(f"{name} training complete.")
        print(f"Training time: {time_taken:.4f} seconds")
        print(f"Training memory: {memory_used:.4f} MiB")
        
        # Save model
        model_path = f'{model_save_prefix}_model_{name}.keras'
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        trained_models[name] = model
    
    # Load and preprocess test data
    print("\nLoading and preprocessing test data...")
    test = pd.read_csv(test_path)
    
    y_test = test['label']
    y_test = label_encoder.transform(y_test)
    
    test = test.drop('label', axis=1)
    test['protocol'] = protocol_encoder.transform(test['protocol'])
    test = test.drop('Unnamed: 0', axis=1)
    
    # Process payload data
    payload_length = []
    count_list = []
    print("Processing payload data...")
    
    for i, row in test.iterrows():
        count_list.append(count(row['payload']))
        payload_length.append(len(list(str(row['payload']).split(','))[:-1]))
    
    test = test.drop('payload', axis=1)
    test['payload_length'] = payload_length
    
    # Apply PCA transformation
    pca_loaded = joblib.load(pca_path)
    reduced_payloads = []
    
    for i in count_list:
        reduced_vector = pca_loaded.transform(i.reshape(1, -1))
        reduced_payloads.append(reduced_vector)
    
    test['reduced_payload'] = reduced_payloads
    array_df = pd.DataFrame(test['reduced_payload'].apply(lambda x: x.flatten()).tolist(), 
                           columns=[f'{i+1}' for i in range(12)])
    test = test.drop('reduced_payload', axis=1)
    test = pd.concat([test, array_df], axis=1, join='inner')
    
    # Balance test data
    y_test = pd.Series(y_test)
    desired_count_over = 500
    desired_count_under = 500 
    oversample_strategy = {i: desired_count_over for i in range(len(y_test.value_counts())) 
                          if y_test.value_counts()[i] < desired_count_over}
    undersample_strategy = {i: desired_count_under for i in range(len(y_test.value_counts())) 
                           if y_test.value_counts()[i] > desired_count_under}
    
    smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=1)
    undersample = RandomUnderSampler(sampling_strategy=undersample_strategy)
    pipeline = Pipeline(steps=[('smote', smote), ('undersample', undersample)])
    
    print("Test data before resampling:", y_test.value_counts())
    x_test_resampled, y_test_resampled = pipeline.fit_resample(test, y_test)
    print("Test data after resampling:", y_test_resampled.value_counts())
    
    # Evaluate all models
    results = {}
    
    for name, model in trained_models.items():
        is_proposed = (name == "Proposed_Model")
        accuracy, precision, recall, f1 = evaluate_model(
            model, x_test_resampled, y_test_resampled, name, label_encoder, is_proposed
        )
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

if __name__ == "__main__":
    main()