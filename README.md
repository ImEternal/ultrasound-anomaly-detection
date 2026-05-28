# ultrasound-anomaly-detection
## Overview

This project simulates simplified ultrasound-like sensor signals and applies machine learning techniques to detect anomalies.  
It is inspired by industrial pipeline inspection systems, where identifying defects in sensor data is a critical task for predictive maintenance and structural health monitoring.

The goal is to demonstrate skills in:
- Signal processing
- Feature engineering
- Machine learning (supervised classification)
- Data simulation and modeling

---

## Problem Statement

In industrial inspection systems (e.g. pipeline monitoring), sensor signals often contain noise and hidden defects.  
The task is to automatically distinguish between:

- **Normal signals**
- **Defective / anomalous signals**

This project builds a simplified simulation of this scenario and applies ML methods for classification.

---

##  Methodology

### 1. Signal Simulation
Synthetic 1D time-series signals are generated:
- Normal signals: sinusoidal wave + Gaussian noise
- Defective signals: sinusoidal wave + noise + artificial spikes

### 2. Feature Extraction
Each signal is transformed into statistical features:
- Mean
- Standard deviation
- Maximum value
- Minimum value
- Peak-to-peak amplitude

### 3. Machine Learning Model
A Random Forest classifier is trained to distinguish between normal and defective signals.

---

## Workflow

Signal Generation 
→ Feature Extraction 
→ Train/Test Split 
→ Model Training 
→ Evaluation

---

## Results

The model achieves reasonable classification performance on synthetic data, demonstrating that simple statistical features can already capture anomaly patterns in noisy sensor signals.

Add Matrix here later

---

## Tech Stack

- Python 3.x
- NumPy
- scikit-learn
- Matplotlib (optional for visualization)

---

How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

python src/train.py