# Parkinson's Disease Detection Using Machine Learning

This Jupyter Notebook provides a step-by-step workflow for detecting Parkinson's disease using machine learning techniques. The dataset is analyzed, preprocessed, and used to train a Support Vector Machine (SVM) model. The trained model is then evaluated for accuracy and saved for future use.

## Dataset
The dataset used in this project is `parkinsons.csv`. It includes various biomedical voice measurements to classify individuals into healthy or Parkinson's disease-positive categories.

### Key Columns:
- `status`: Target variable (`1` indicates Parkinson's, `0` indicates healthy).
- Other features: Biomedical voice metrics.

---

## Workflow

### 1. **Data Loading and Exploration**
   - Load the dataset using pandas and inspect the data structure using functions like `.head()`, `.info()`, `.describe()`, etc.
   - Explore the distribution of the target variable (`status`).

### 2. **Data Preprocessing**
   - Separate features (`X`) from the target variable (`Y`).
   - Split the dataset into training and testing sets using `train_test_split` (80-20 split).
   - Standardize the feature set using `StandardScaler`.

### 3. **Model Training**
   - Train an SVM classifier with a linear kernel using the standardized training data.
   - Evaluate the model's accuracy on both training and test datasets using `accuracy_score`.

### 4. **Prediction**
   - Test the model with a sample input to predict whether the individual has Parkinson's disease.
   - Print the prediction result.

### 5. **Model Persistence**
   - Save the trained model using Python's `pickle` library.
   - Demonstrate loading the saved model and making predictions.

---

## Code Snippets
### Import Required Libraries
```python
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle




X = park.drop(columns=['name', 'status'], axis=1)
Y = park['status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)





model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)




training_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))
print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

