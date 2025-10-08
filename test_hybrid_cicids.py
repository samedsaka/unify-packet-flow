# -*- coding: utf-8 -*-


import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Configuration constants
DEFAULT_OVERSAMPLE_COUNT = 500
DEFAULT_UNDERSAMPLE_COUNT = 500
FLOW_OVERSAMPLE_COUNT = 2000
FLOW_UNDERSAMPLE_COUNT = 10000
RANDOM_STATE = 42

# CICIDS attack type mappings
CICIDS_PACKET_LEVEL_LABELS = [0, 7, 8, 11, 12, 13, 14]
CICIDS_FLOW_LEVEL_LABELS = [1, 2, 3, 4, 5, 6, 9, 10]

# File paths 
PACKET_TEST_DATASET_PATH = "merged_headerand12Payload_cicids_test.csv"
CICIDS_DATASET_PATH = "CICIDS2017\\MachineLearningCVE\\"
PACKET_MODEL_PATH = 'cicids_model_Proposed_Model.keras'
FLOW_MODEL_PATH = "cicids_flow_cnn.keras"


def balance(x, y, desired_count_over, desired_count_under):
    y = pd.Series(y)
    oversample_strategy = {i: desired_count_over for i in range(len(y.value_counts())) if y.value_counts()[i] < desired_count_over}
    undersample_strategy = {i: desired_count_under for i in range(len(y.value_counts())) if y.value_counts()[i] > desired_count_under}

    # Create the SMOTE and RandomUnderSampler objects
    smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=1)
    undersample = RandomUnderSampler(sampling_strategy=undersample_strategy)

    # Combine SMOTE and RandomUnderSampler in a pipeline
    pipeline = Pipeline(steps=[('smote', smote), ('undersample', undersample)])

    # Print class distribution before resampling
    print("Before resampling:", y.value_counts())
    print()

    x_resampled, y_resampled = pipeline.fit_resample(x, y)

    # Print class distribution after resampling
    print("After resampling:", y_resampled.value_counts())
    return x_resampled, y_resampled
    
    
def plot_percentage_occurrence_histogram(data_array, title="Histogram of Percentage Occurrence"):
    # Calculate unique values and their counts
    unique, counts = np.unique(data_array, return_counts=True)

    # Calculate percentage occurrence
    total_count = data_array.size
    percentages = (counts / total_count) * 100

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, percentages, tick_label=unique)
    plt.xlabel('Unique Values')
    plt.ylabel('Percentage Occurrence (%)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add numbers above bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%',
                 ha='center', va='bottom')

    plt.show()


def load_and_preprocess_packet_data():
    """Load and preprocess packet-level test dataset."""
    print("Loading packet test dataset...")
    packet_test_dataset = pd.read_csv(PACKET_TEST_DATASET_PATH)
    
    packet_X = packet_test_dataset.iloc[:, :-1].values
    packet_y = packet_test_dataset.iloc[:, -1].values
    
    plot_percentage_occurrence_histogram(packet_y, "Packet Dataset Class Distribution")
    packet_X, packet_y = balance(packet_X, packet_y, DEFAULT_OVERSAMPLE_COUNT, DEFAULT_UNDERSAMPLE_COUNT)
    
    return packet_X, packet_y


def load_and_preprocess_flow_data():
    """Load and preprocess flow-level CICIDS dataset."""
    print("Loading CICIDS flow dataset...")
    cicids_files = glob.glob(CICIDS_DATASET_PATH + "*.csv")
    df = pd.concat([pd.read_csv(file) for file in cicids_files], ignore_index=True)
    
    # Clean label names
    label_mappings = {
        'Web Attack � Brute Force': 'WA-Brute Force',
        'Web Attack � XSS': 'WA-XSS',
        'Web Attack � Sql Injection': 'WA-SqlInjection'
    }
    
    for old_label, new_label in label_mappings.items():
        df[' Label'] = df[' Label'].apply(lambda x: new_label if x == old_label else x)
    
    # Plot class distribution
    df[' Label'].value_counts().plot(kind='bar')
    plt.title("Flow Dataset Class Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Separate features and labels
    flow_X = df.drop([' Label'], axis=1)
    flow_y = df[[' Label']]
    
    # Encode labels
    label_encoder = LabelEncoder()
    flow_y[' Label'] = label_encoder.fit_transform(flow_y[' Label'])
    
    # Handle missing values and scale features
    flow_X = np.nan_to_num(flow_X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    flow_X = scaler.fit_transform(flow_X)
    
    return flow_X, flow_y, label_encoder


def prepare_flow_test_data(flow_X, flow_y):
    """Prepare flow test data with balancing."""
    print("Preparing flow test data...")
    _, flow_x_test, _, flow_y_test = train_test_split(
        flow_X, flow_y, test_size=0.2, random_state=RANDOM_STATE
    )
    flow_x_test, flow_y_test = balance(
        flow_x_test, flow_y_test.squeeze(), FLOW_OVERSAMPLE_COUNT, FLOW_UNDERSAMPLE_COUNT
    )
    flow_y_test = to_categorical(flow_y_test)
    return flow_x_test, flow_y_test


def load_models():
    """Load pre-trained packet and flow models."""
    print("Loading pre-trained models...")
    packet_model = load_model(PACKET_MODEL_PATH)
    flow_model = load_model(FLOW_MODEL_PATH)
    return packet_model, flow_model


def hybrid_prediction(packet_model, flow_model, packet_X, packet_y, flow_x_test, flow_y_test):
    print("Starting hybrid prediction...")
    y_pred_list = []
    
    # Get flow attack indices for random sampling
    flow_attacks_indexes = [
        i for i, value in enumerate(flow_y_test.argmax(axis=1)) 
        if value in CICIDS_FLOW_LEVEL_LABELS
    ]
    
    total_samples = len(packet_X)
    
    # Iterate through the packet dataset
    for i in range(total_samples):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing sample {i}/{total_samples}")
            
        packet_sample = packet_X[i]
        packet_label = packet_y[i]
        
        packet_sample = pd.DataFrame(packet_sample).transpose()
        
        # Predict with the packet model
        packet_pred = packet_model.predict([
            packet_sample.iloc[:, :5], 
            packet_sample.iloc[:, 5:]
        ], verbose=0)
        packet_pred_class = np.argmax(packet_pred, axis=1)

        # Check if the prediction is packet-level attack
        if packet_pred_class in CICIDS_PACKET_LEVEL_LABELS:
            y_pred_list.append((packet_pred_class.item(), packet_label, 'P'))
        else:
            # Use flow model for flow-level attacks
            if flow_attacks_indexes:  # Ensure there are flow attack samples
                # Random sampling from flow dataset
                random_index = random.choice(range(len(flow_x_test)))
                flow_x_sample = flow_x_test[random_index]
                flow_x_sample = pd.DataFrame(flow_x_sample).transpose()
                flow_y_sample = flow_y_test[random_index]

                # Predict with the flow model
                flow_pred = flow_model.predict(
                    np.expand_dims(flow_x_sample, axis=2), verbose=0
                )
                flow_pred_class = flow_pred.argmax(axis=1)
                y_pred_list.append((
                    flow_pred_class.item(), 
                    flow_y_sample.argmax(axis=0), 
                    'F'
                ))
    
    return y_pred_list


def evaluate_and_save_results(y_pred_list, label_encoder):
    """Evaluate hybrid model results and save outputs."""
    print("Evaluating results...")
    
    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(y_pred_list, columns=['Prediction', 'True Label', 'Model'])
    
    # Save results
    y_pred_df.to_csv('cicids_hybrid_results.csv', index=False)
    print("Results saved to 'cicids_hybrid_results.csv'")
    
    # Create confusion matrix
    labels = label_encoder.classes_
    cm = confusion_matrix(
        y_pred_df['True Label'], 
        y_pred_df['Prediction'], 
        normalize='true'
    )
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=80, include_values=False)
    plt.title("Hybrid Model Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
    # Generate and save classification report
    report = classification_report(y_pred_df['True Label'], y_pred_df['Prediction'])
    print("\nClassification Report:")
    print(report)
    
    with open('cicids_classification_report_hybrid.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to 'cicids_classification_report_hybrid.txt'")


def main():
    """Main execution function."""
    try:
        # Load and preprocess data
        packet_X, packet_y = load_and_preprocess_packet_data()
        flow_X, flow_y, label_encoder = load_and_preprocess_flow_data()
        
        # Prepare flow test data
        flow_x_test, flow_y_test = prepare_flow_test_data(flow_X, flow_y)
        
        # Load models
        packet_model, flow_model = load_models()
        
        # Perform hybrid prediction
        y_pred_list = hybrid_prediction(
            packet_model, flow_model, packet_X, packet_y, flow_x_test, flow_y_test
        )
        
        # Evaluate and save results
        evaluate_and_save_results(y_pred_list, label_encoder)
        
        print("Hybrid classification completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()


