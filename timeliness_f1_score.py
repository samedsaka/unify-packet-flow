# -*- coding: utf-8 -*-

import pandas as pd

# Read 2 result files from hybrid approach
unsw_hybrid_results = pd.read_csv("unsw_hybrid_results.csv")
cicids_hybrid_results = pd.read_csv("cicids_hybrid_results.csv")

# Display model distribution histograms
unsw_hybrid_results['model'].hist()
cicids_hybrid_results['model'].hist()

# Calculate Delta_t (timeliness factor)
unsw_model_counts = unsw_hybrid_results['model'].value_counts()
cicids_model_counts = cicids_hybrid_results['model'].value_counts()

epsilon = 0.01

# Calculate Nf / Ntotal for each dataset
unsw_nf_ratio = unsw_model_counts.iloc[0] / (unsw_model_counts.iloc[0] + unsw_model_counts.iloc[1])
cicids_nf_ratio = cicids_model_counts.iloc[0] / (cicids_model_counts.iloc[0] + cicids_model_counts.iloc[1])

# Apply the delta formula: Δt = max(0, Nf / Ntotal - ε)
unsw_delta = max(0, unsw_nf_ratio - epsilon)
cicids_delta = max(0, cicids_nf_ratio - epsilon)

# F1 scores for hybrid approach
unsw_hybridF1 = 0.77
cicids_hybridF1 = 0.98

# Alpha parameter for TFS calculation
alpha = 0.5

# Calculate TFS (Timeliness F1 Score) for both datasets
tfs_unsw_hybrid = (alpha * unsw_hybridF1) + ((1 - alpha) * (1 - unsw_delta))
tfs_cicids_hybrid = (alpha * cicids_hybridF1) + ((1 - alpha) * cicids_delta)

print(f'TFS UNSW Hybrid: {tfs_unsw_hybrid:.4f}')
print(f'TFS CICIDS Hybrid: {tfs_cicids_hybrid:.4f}')
print(f'UNSW Delta: {unsw_delta:.4f}')
print(f'CICIDS Delta: {cicids_delta:.4f}')

















