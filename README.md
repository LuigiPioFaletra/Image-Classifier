# Python Project - Image Classification with MLP Neural Networks

This repository contains a Python project for **image classification**, focused on predicting both **direction** and **emotion** from grayscale images.  
The work was carried out as part of the *Advanced Programming / Machine Learning* coursework.

The project implements:
- Dataset preparation  
- Image preprocessing (RGB conversion, scaling, reshaping)  
- Train/test set generation for multiple image sizes  
- Multi-layer Perceptron classifiers (MLPClassifier)  
- Evaluation of 10 different neural network models for **direction** and **emotion** prediction  

---

## Repository Structure

```
main_repository/
│
├── faces.zip
├── LICENSE
├── Progetto.ipynb
└── README.md
```

---

## Project Overview

The goal is to build classification models capable of predicting:
- **Direction** (e.g., left, right, center…)  
- **Emotion** (e.g., happy, sad, neutral…)  

using neural networks trained on small grayscale images of different sizes.

The project includes:
- Image loading and labeling  
- Feature extraction via RGB sequences  
- Data normalization  
- Dataset reshaping for MLP input  
- Training and evaluation of multiple MLP models with different hyperparameters  

---

## Workflow Summary

### **1. Function Definitions**
All helper functions for loading, labeling, splitting, converting, and reshaping images are defined at the beginning of the notebook.

### **2. Image Importing & Labeling**
- Images are loaded from the dataset folder  
- Each image receives two labels:  
  - **Direction**  
  - **Emotion**  
- Labeled images are stored into two separate lists:
  - List for direction classification  
  - List for emotion classification  

### **3. Creation of Label Arrays**
For each image size:
- Arrays containing the corresponding labels are generated  
- Labels are stored separately for:
  - Direction  
  - Emotion  

### **4. RGB Extraction**
Each image is:
- Converted into a sequence of RGB pixel values  
- Stored into dedicated arrays  

### **5. Preprocessing & Normalization**
- Pixel values are converted back into arrays  
- Normalized by dividing by **255**  
- Reshaped into 4D arrays:  
(num_images, width, height, 1)

sql
Copia codice
- Label arrays are reshaped accordingly  
- Normalized image arrays are extracted from their lists into standalone arrays  

### **6. Train/Test Split**
For each image size and for both tasks (direction/emotion):
- Independent training and testing sets are generated  

### **7. Final Reshape for MLP**
MLP requires a 2D input, so all images are reshaped into:
(num_images, width * height * 1)

markdown
Copia codice

### **8. Model Training & Evaluation**
For each task (direction, emotion), **10 MLP classifiers** are trained with different hyperparameters.

For each model:
- Initialization of `MLPClassifier`  
- Training (`fit`)  
- Prediction  
- Accuracy score calculation  

---

## MLP Models Implemented

Below are the **10 models**, applied identically for both direction and emotion classification.

| Model | Hidden Layers | Activation | Solver | Batch Size | Learning Rate | Max Iter | Tolerance | Notes |
|------|----------------|------------|--------|------------|----------------|----------|----------|--------|
| **1** | 16 | relu | adam | 50 | constant | 400 | 0.00001 | verbose=True |
| **2** | 16 | identity | sgd | 50 | constant | 500 | 0.0001 | verbose=True |
| **3** | 32 | relu | adam | 150 | adaptive | 200 | 0.00001 | verbose=True |
| **4** | 32 | tanh | adam | 150 | constant | 300 | 0.00001 | verbose=True |
| **5** | 64 | relu | sgd | 250 | constant | 900 | 0.00001 | verbose=True |
| **6** | 64 | logistic | adam | 250 | adaptive | 700 | 0.0001 | verbose=True |
| **7** | 128 | relu | sgd | 400 | adaptive | 1000 | 0.0001 | verbose=True |
| **8** | 128 | identity | adam | 400 | invscaling | 200 | 0.00001 | verbose=True |
| **9** | 256 | relu | adam | 500 | invscaling | 200 | 0.0001 | verbose=True |
| **10** | 256 | identity | adam | 500 | adaptive | 300 | 0.0001 | verbose=True |

---

## Technologies & Libraries Used

- **Python**
- **NumPy**
- **Scikit-learn**
- **Pillow** (for image processing)
- **Matplotlib** (visualization)
- **Jupyter Notebook**

---

## Execution

Open the notebook:

```bash
jupyter notebook implementation.ipynb
```

Run all cells to:
- Preprocess the images
- Prepare training/testing sets
- Train the 10 neural network models
- Display accuracy scores for both direction & emotion classifiers

---

## Results

The notebook prints:
1. Accuracy for each of the 10 models
2. Comparative performance
3. Differences between direction and emotion classification
4. Observations on how accuracy changes with:
  - Image size
  - Model complexity
  - Activation function
  - Solver
  - Batch size

---

## Notes

- The file **faces.zip** must be extracted into a folder named **faces** located in the root directory of the repository.
- Execution time may be long due to 20 total models (10 for each classification task).
- Higher hidden layer size tends to increase training time.
- MLPClassifier requires flattened image input, hence the multiple reshape steps.

---

### License

This project is licensed under the terms of the MIT license. You can find the full license in the `LICENSE` file.
