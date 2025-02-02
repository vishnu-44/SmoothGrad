# SmoothGrad and SmoothGrad with Integrated Gradients

This repository contains implementations of **SmoothGrad** and an extended version of **SmoothGrad with Integrated Gradients (IG)**. The code is organized into two main folders, each containing the necessary files for the respective methods. Additionally, the datasets used in the experiments are stored in a separate `datasets` folder.

---

## Repository Structure

---

## Folder Descriptions

### 1. **SmoothGrad**
This folder contains the implementation of the standard **SmoothGrad** method. SmoothGrad is a technique for visualizing feature importance in neural networks by adding noise to the input and averaging the resulting gradients.

- **Files**:
  - `SmoothGrad.py`: A Python script for running SmoothGrad.
  - `SmoothGrad.ipynb`: A Jupyter Notebook demonstrating the usage of SmoothGrad with examples.

---

### 2. **SmoothGrad_Integrated_Gradients**
This folder contains the implementation of an extended version of SmoothGrad that combines it with **Integrated Gradients (IG)**. Integrated Gradients is a method for attributing predictions to input features by integrating gradients along a path from a baseline to the input.

- **Files**:
  - `SmoothGrad_IG.ipynb`: A Jupyter Notebook demonstrating the combined SmoothGrad and Integrated Gradients approach.

---

### 3. **Datasets**
This folder contains the datasets used in the experiments for both SmoothGrad and SmoothGrad with Integrated Gradients and feel free to use any.

- **Files**:
  - `dataset1.csv`: Example dataset 1.
  - `dataset2.csv`: Example dataset 2.

---

## Usage

### Running the Code
1. **SmoothGrad**:
   - Open the `SmoothGrad.ipynb` notebook or run the `SmoothGrad.py` script.
   - Follow the instructions in the notebook/script to load the dataset and generate visualizations.

2. **SmoothGrad with Integrated Gradients**:
   - Open the `SmoothGrad_IG.ipynb` notebook.
   - Follow the instructions to combine SmoothGrad with Integrated Gradients and generate visualizations.


