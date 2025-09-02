# OWZP-Transformer

This is for the paper "Enhancing tropical cyclone track and intensity predictions with the OWZP-Transformer model".

This repository contains the implementation of the OWZP-Transformer, a deep learning model developed to improve the prediction of tropical cyclone (TC) track, intensity, and rapid intensification (RI) events over the Western North Pacific.

The model incorporates 15 predictors, including the Okubo–Weiss–Zeta Parameter (OWZP) as a structural factor, and leverages a Transformer-based architecture with dual encoders and a composite loss function. Our experiments demonstrate significant improvements in short-term TC forecasts compared with existing deep learning models.

**Repository Structure**

'Data_processing.py'
Prepares input datasets from IBTrACS best-track records and ERA5 reanalysis. Includes feature engineering (e.g., steering flow, OWZP, VWS, RH, SH), sliding-window construction, scaling, and dataset splits into training/validation/test sets.

Train_test.py
Defines the OWZP-Transformer architecture, multi-task loss (Haversine for track, RMSE for intensity), and training/testing loops. Evaluates model skill using RMSE, MAE, R², and bias.

Optimization.py
Implements hyperparameter tuning with Optuna, including embedding dimension, attention heads, feedforward size, number of encoder/decoder layers, learning rate, dropout, and batch size.

Ablation_plot.py
Produces ablation study figures showing the effect of removing predictor categories (structural, environmental, gradient). Also compares OWZP-Transformer against baseline models.

Feature_importance.py
Explains model predictions using Captum’s DeepLIFT and DeepLiftShap methods. Generates feature attribution plots across predictors for intensity, pressure, track, and RI.

TC_RI_nonRI.py
Identifies cyclones undergoing rapid intensification (RI) or non-RI events based on test data. Prepares SID lists and event classification for further visualization.

RI_nonRI_plot.py
Plots case studies of detected RI, missed RI, and non-RI cyclones. Compares true vs. predicted tracks and intensities, with cartographic visualization.
