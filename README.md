# Unify Packet & Flow Classification

Collection of scripts that implement preprocessing, dimensionality reduction, model training, and hybrid evaluation components for a proposed packet+flow hybrid classification system.

## What is included

- Packet-level preprocessing (payload trimming, byte counts, train/test split)
- Dimensionality reduction experiments (PCA, NMF) on payload-count vectors
- Packet-level models: NN, CNN, and a proposed dual-path model
- Flow-level models: flow DNN/CNN training for CICIDS/UNSW datasets
- Hybrid evaluation scripts that combine packet- and flow-level predictions
- Timeliness-F1 (TFS) metric calculation for hybrid results

- CICIDS flow CSVs: expected under `CICIDS2017\MachineLearningCVE\*.csv` ([CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html))
- UNSW flow CSVs: `UNSW-NB15\OneDrive_2024-02-16\CSV Files\...` ([UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset))
- Packet CSVs: `Payload_data_UNSW.csv` and `Payload_data_CICIDS2017.csv` ([Payload-Byte Repository](https://github.com/Yasir-ali-farrukh/Payload-Byte))
