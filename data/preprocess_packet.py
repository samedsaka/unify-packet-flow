import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os


def load_packet_data(file_path):
    print(f"Loading data from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the CSV file
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    
    return data


def extract_components(data):
    print("Extracting payload, header, and label components...")
    
    # Extract payload data (first 1500 columns)
    payload_data = data.iloc[:, :1500]
    
    # Extract header info (columns 1500-1503: ttl, total_len, protocol, t_delta)
    header_columns = ['ttl', 'total_len', 'protocol', 't_delta']
    header_data = data.iloc[:, 1500:1504]
    header_data.columns = header_columns
    
    # Extract labels (last column)
    labels = data.iloc[:, -1]
    
    print(f"Payload shape: {payload_data.shape}")
    print(f"Header shape: {header_data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return payload_data, header_data, labels


def trim_leading_zeros(payload_row):
    # Convert to numpy array and find first non-zero index
    payload_array = payload_row.values
    
    # Find first non-zero element
    non_zero_indices = np.nonzero(payload_array)[0]
    
    if len(non_zero_indices) == 0:
        # All zeros, return empty array
        return np.array([])
    
    # Return payload from first non-zero element onwards
    first_non_zero = non_zero_indices[0]
    return payload_array[first_non_zero:]


def count_byte_occurrences(trimmed_payload):
    # Initialize count array for bytes 0-255
    byte_counts = np.zeros(256, dtype=int)
    
    # Count occurrences of each byte value
    if len(trimmed_payload) > 0:
        unique_values, counts = np.unique(trimmed_payload.astype(int), return_counts=True)
        
        # Fill the count array
        for value, count in zip(unique_values, counts):
            if 0 <= value <= 255:  # Ensure valid byte range
                byte_counts[value] = count
    
    return byte_counts


def preprocess_payload(payload_data):
    print("Preprocessing payload data...")
    print("- Trimming leading zeros")
    print("- Counting byte occurrences")
    
    processed_payloads = []
    packet_lengths = []
    
    # Process each payload row iteratively
    for idx, row in payload_data.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing row {idx}/{len(payload_data)}")
        
        # Trim leading zeros
        trimmed_payload = trim_leading_zeros(row)
        
        # Calculate packet length after trimming
        packet_length = len(trimmed_payload)
        packet_lengths.append(packet_length)
        
        # Count byte occurrences
        byte_counts = count_byte_occurrences(trimmed_payload)
        processed_payloads.append(byte_counts)
    
    # Convert to DataFrame with column names for each byte value
    byte_columns = [f'byte_{i}_count' for i in range(256)]
    processed_df = pd.DataFrame(processed_payloads, columns=byte_columns)
    
    # Add packet length column
    processed_df['packet_len'] = packet_lengths
    
    print(f"Processed payload shape: {processed_df.shape}")
    return processed_df


def merge_features(header_data, processed_payload, labels):
    print("Merging header info, processed payload, and labels...")
    
    # Reset indices to ensure proper alignment
    header_data = header_data.reset_index(drop=True)
    processed_payload = processed_payload.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    
    # Combine all features
    combined_data = pd.concat([header_data, processed_payload], axis=1)
    combined_data['label'] = labels
    
    print(f"Combined data shape: {combined_data.shape}")
    return combined_data


def split_and_save_data(combined_data, dataset_name):
    print(f"Splitting {dataset_name} data into train/test sets...")
    
    # Separate features and labels
    X = combined_data.drop('label', axis=1)
    y = combined_data['label']
    
    # Stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Train label distribution: {y_train.value_counts().to_dict()}")
    print(f"Test label distribution: {y_test.value_counts().to_dict()}")
    
    # Combine features and labels for saving
    train_data = X_train.copy()
    train_data['label'] = y_train
    
    test_data = X_test.copy()
    test_data['label'] = y_test
    
    # Save to CSV files
    train_filename = f'x_train_counted_{dataset_name}.csv'
    test_filename = f'x_test_counted_{dataset_name}.csv'
    
    print(f"Saving train data to: {train_filename}")
    train_data.to_csv(train_filename, index=False)
    
    print(f"Saving test data to: {test_filename}")
    test_data.to_csv(test_filename, index=False)


def process_dataset(file_path, dataset_name):
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    try:
        # Step 1: Load data
        data = load_packet_data(file_path)
        
        # Step 2: Extract components
        payload_data, header_data, labels = extract_components(data)
        
        # Step 3: Preprocess payload
        processed_payload = preprocess_payload(payload_data)
        
        # Step 4: Merge all features
        combined_data = merge_features(header_data, processed_payload, labels)
        
        # Step 5: Split and save data
        split_and_save_data(combined_data, dataset_name)
        
        print(f"\n✓ Successfully processed {dataset_name} dataset")
        
    except Exception as e:
        print(f"\n✗ Error processing {dataset_name} dataset: {str(e)}")
        raise


def main():
    print("Packet Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Dataset file paths
    datasets = [
        ('Payload_data_UNSW.csv', 'unsw'),
        ('Payload_data_CICIDS2017.csv', 'cicid')
    ]
    
    # Process each dataset iteratively
    for file_path, dataset_name in datasets:
        process_dataset(file_path, dataset_name)
    
    print(f"\n{'='*50}")
    print("All datasets processed successfully!")
    print("Output files:")
    print("- x_train_counted_unsw.csv")
    print("- x_test_counted_unsw.csv")
    print("- x_train_counted_cicid.csv")
    print("- x_test_counted_cicid.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



