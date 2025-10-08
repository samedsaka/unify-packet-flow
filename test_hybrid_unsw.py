
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Configuration constants
PACKET_DATA_PATH = "merged_headerand12Payload_unsw_test.csv"
FLOW_TRAIN_PATH = "UNSW-NB15\\OneDrive_2024-02-16\\CSV Files\\Training and Testing Sets\\UNSW_NB15_training-set.csv"
FLOW_TEST_PATH = "UNSW-NB15\\OneDrive_2024-02-16\\CSV Files\\Training and Testing Sets\\UNSW_NB15_testing-set.csv"
PACKET_MODEL_PATH = "unsw_model_Proposed_model_new.keras"
FLOW_MODEL_PATH = "unsw_flow_cnn.keras"

# Attack type mappings for UNSW-NB15 dataset
UNSW_PACKET_LEVEL_LABELS = [8, 9, 6, 0]  # Packet-level detectable attacks
UNSW_FLOW_LEVEL_LABELS = [7, 2, 5, 3, 4]  # Flow-level detectable attacks

def balance_dataset(x, y, desired_count_over, desired_count_under):
    y = pd.Series(y)
    
    # Define sampling strategies
    oversample_strategy = {
        i: desired_count_over for i in range(len(y.value_counts())) 
        if y.value_counts()[i] < desired_count_over
    }
    undersample_strategy = {
        i: desired_count_under for i in range(len(y.value_counts())) 
        if y.value_counts()[i] > desired_count_under
    }

    # Create sampling pipeline
    smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=1)
    undersample = RandomUnderSampler(sampling_strategy=undersample_strategy)
    pipeline = Pipeline(steps=[('smote', smote), ('undersample', undersample)])

    print("Before resampling:", y.value_counts())
    x_resampled, y_resampled = pipeline.fit_resample(x, y)
    print("After resampling:", y_resampled.value_counts())
    
    return x_resampled, y_resampled
    
    
def plot_percentage_occurrence_histogram(data_array):
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
    plt.title('Histogram of Percentage Occurrence')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add numbers above bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%',
                 ha='center', va='bottom')

    plt.show()


def main():
    """Main function to execute hybrid model testing."""
    # Load and prepare packet data
    print("Loading packet test dataset...")
    packet_test_dataset = pd.read_csv(PACKET_DATA_PATH)
    packet_X = packet_test_dataset.iloc[:, :-1].values
    packet_y = packet_test_dataset.iloc[:, -1].values

    # Balance packet dataset
    packet_X, packet_y = balance_dataset(packet_X, packet_y, 500, 500)
    plot_percentage_occurrence_histogram(packet_y)

    # Load and prepare flow data
    print("Loading flow datasets...")
    train_flow = pd.read_csv(FLOW_TRAIN_PATH)
    test_flow = pd.read_csv(FLOW_TEST_PATH)
    flow_dataset = pd.concat((train_flow, test_flow), axis=0)

    # Encode categorical features
    print("Encoding categorical features...")
    label_encoders = {}
    for column in ['proto', 'service', 'state', 'attack_cat']:
        le = LabelEncoder()
        flow_dataset[column] = le.fit_transform(flow_dataset[column])
        label_encoders[column] = le

    # Prepare features and target variable
    flow_X = flow_dataset.drop(['attack_cat', 'label', 'id'], axis=1)
    flow_y = flow_dataset['attack_cat']

    # Handle missing and infinite values
    flow_X = np.nan_to_num(flow_X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    scaler = StandardScaler()
    flow_X = scaler.fit_transform(flow_X)

    # Split and balance flow dataset
    print("Preparing flow test data...")
    _, flow_x_test, _, flow_y_test = train_test_split(flow_X, flow_y, test_size=0.2, random_state=42)
    flow_x_test, flow_y_test = balance_dataset(flow_x_test, flow_y_test, 1000, 1000)
    plot_percentage_occurrence_histogram(flow_y_test)

    # Convert labels to categorical format
    flow_y_test = to_categorical(flow_y_test)

    # Load pre-trained models
    print("Loading trained models...")
    packet_model = load_model(PACKET_MODEL_PATH)
    flow_model = load_model(FLOW_MODEL_PATH)

    # Get indices of flow-level attacks for sampling
    flow_attacks_indexes = [
        i for i, value in enumerate(flow_y_test.argmax(axis=1)) 
        if value in UNSW_FLOW_LEVEL_LABELS
    ]

    # Initialize prediction storage
    y_pred_list = []

    # Hybrid prediction loop
    print("Starting hybrid predictions...")
    for i, (packet_sample, packet_label) in enumerate(zip(packet_X, packet_y)):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(packet_X)}")
        
        # Prepare packet sample for prediction
        packet_sample_df = pd.DataFrame(packet_sample).transpose()
        
        # Predict with packet model (using multi-input architecture)
        packet_pred = packet_model.predict([
            packet_sample_df.iloc[:, :5], 
            packet_sample_df.iloc[:, 5:]
        ], verbose=0)
        packet_pred_class = np.argmax(packet_pred, axis=1)

        # Check if prediction is packet-level detectable
        if packet_pred_class in UNSW_PACKET_LEVEL_LABELS:
            y_pred_list.append((packet_pred_class.item(), packet_label, 'P'))
        else:
            # Use flow model for flow-level attacks
            if len(flow_x_test) > 0:
                # Select random flow sample for prediction
                random_index = random.choice(range(min(10000, len(flow_x_test))))
                flow_x_sample = flow_x_test[random_index]
                flow_y_sample = flow_y_test[random_index]
                
                # Prepare flow sample and predict
                flow_x_sample_df = pd.DataFrame(flow_x_sample).transpose()
                flow_pred = flow_model.predict(
                    np.expand_dims(flow_x_sample_df, axis=2), verbose=0
                )
                flow_pred_class = flow_pred.argmax(axis=1)
                
                y_pred_list.append((
                    flow_pred_class.item(), 
                    flow_y_sample.argmax(axis=0), 
                    'F'
                ))

    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(y_pred_list, columns=['Prediction', 'True Label', 'Model'])

    # Save predictions to CSV
    print("Saving results...")
    y_pred_df.to_csv('unsw_hybrid_results_cleaned.csv', index=False)

    # Generate evaluation metrics
    print("Generating evaluation metrics...")
    labels = label_encoders['attack_cat'].classes_

    # Create and display confusion matrix
    cm = confusion_matrix(y_pred_df['True Label'], y_pred_df['Prediction'], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=80, include_values=False)
    plt.title("Hybrid Model Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Generate and display classification report
    print("\nClassification Report:")
    report = classification_report(y_pred_df['True Label'], y_pred_df['Prediction'])
    print(report)

    # Save classification report to file
    with open('unsw_classification_report_hybrid.txt', 'w') as f:
        f.write("UNSW-NB15 Hybrid Model Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print("Analysis complete!")


if __name__ == "__main__":
    main()