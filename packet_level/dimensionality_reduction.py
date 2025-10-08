# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import mean_squared_error
import psutil
import os
import matplotlib.pyplot as plt


def measure_performance(model, transform_func, inverse_transform_func, data, n_components, method_name):
    """Measure performance metrics for dimensionality reduction methods"""
    process = psutil.Process(os.getpid())
    
    # Record initial state
    start_time = time.time()
    mem_usage_before = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    
    # Fit and transform the model
    transformed_data = transform_func(model, data)
    
    # Record final state
    elapsed_time = time.time() - start_time
    mem_usage_after = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    
    # Reconstruct the data
    reconstructed_data = inverse_transform_func(model, transformed_data)
    
    # Calculate reconstruction error (MSE)
    reconstruction_error = mean_squared_error(data, reconstructed_data)
    memory_usage = mem_usage_after - mem_usage_before
    
    return {
        'method': method_name,
        'n_components': n_components,
        'reconstruction_error': reconstruction_error,
        'time': elapsed_time,
        'memory': memory_usage
    }


def pca_transform(model, data):
    """PCA transform function"""
    return model.transform(data)


def pca_inverse_transform(model, transformed_data):
    """PCA inverse transform function"""
    return model.inverse_transform(transformed_data)


def nmf_transform(model, data):
    """NMF transform function"""
    return model.transform(data)


def nmf_inverse_transform(model, transformed_data):
    """NMF inverse transform function"""
    return np.dot(transformed_data, model.components_)


def analyze_dataset(file_path, dataset_name, n_components_range=range(6, 32, 2), max_samples=500000):
    """Analyze a dataset using PCA and NMF"""
    print(f"\n=== Analyzing {dataset_name} ===")
    
    # Load and preprocess data
    try:
        df = pd.read_csv(file_path, header=None)
        print(f"Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
    # Normalize data
    df = df / np.max(df)
    
    # Limit samples if needed
    if len(df) > max_samples:
        df = df[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # Store results
    results = []
    
    # PCA Analysis
    print("\nRunning PCA analysis...")
    for n_comp in n_components_range:
        print(f"  Processing PCA with {n_comp} components...")
        try:
            pca_model = PCA(n_components=n_comp, random_state=42)
            pca_model.fit(df)
            
            result = measure_performance(
                pca_model, pca_transform, pca_inverse_transform, 
                df, n_comp, 'PCA'
            )
            results.append(result)
        except Exception as e:
            print(f"    Error with PCA {n_comp} components: {e}")
    
    # NMF Analysis
    print("\nRunning NMF analysis...")
    for n_comp in n_components_range:
        print(f"  Processing NMF with {n_comp} components...")
        try:
            nmf_model = NMF(n_components=n_comp, init='random', random_state=42, max_iter=1000)
            nmf_model.fit(df)
            
            result = measure_performance(
                nmf_model, nmf_transform, nmf_inverse_transform, 
                df, n_comp, 'NMF'
            )
            results.append(result)
        except Exception as e:
            print(f"    Error with NMF {n_comp} components: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def main():
    # File paths
    cicids_file = "cicids_payload_counted.csv"
    unsw_file = "unsw_payload_counted.csv"
    
    # Component range
    n_components_range = range(6, 32, 2)
    
    # Analyze CICIDS dataset
    print("Starting analysis...")
    cicids_results = analyze_dataset(cicids_file, "CICIDS", n_components_range)
    
    if cicids_results is not None:
        # Save results
        cicids_results.to_csv('cicids_payload_reduction_results.csv', index=False)
        print(f"\nCICIDS results saved to: cicids_payload_reduction_results.csv")
        
    
    # Analyze UNSW dataset
    unsw_results = analyze_dataset(unsw_file, "UNSW", n_components_range)
    
    if unsw_results is not None:
        # Save results
        unsw_results.to_csv('unsw_payload_reduction_results.csv', index=False)
        print(f"\nUNSW results saved to: unsw_payload_reduction_results.csv")


if __name__ == "__main__":
    main()