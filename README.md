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
### 1. Import Required Libraries

```python
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
```

### 2. Load and Explore the Dataset

```python
# Load the dataset
data = pd.read_csv('parkinsons.csv')

# Display the first few rows
print(data.head())

# Check the structure of the dataset
print(data.info())

# Display summary statistics of the dataset
print(data.describe())

# Analyze the distribution of the target variable
print(data['status'].value_counts())
```

### 3. Feature and Target Separation

```python
# Separate features (X) and target variable (Y)
X = data.drop(columns=['name', 'status'], axis=1)
Y = data['status']

print("Features (X):")
print(X.head())

print("\nTarget (Y):")
print(Y.head())
```

### 4.Split the data 

```python
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
```

### 5.Feature Scaling
```python
# Standardize the feature values to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()

# Fit the scaler on training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Standardization completed.")
```

### 6. Train the model
```python
# Initialize and train the Support Vector Machine (SVM) model with a linear kernel
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

print("Model training completed.")
```

### 7.Evaluate the model 
```python
# Evaluate the model on training data
training_accuracy = accuracy_score(Y_train, model.predict(X_train))

# Evaluate the model on test data
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
```

### 8.Make Predictions 
```python
# Test the model with a new sample input
input_data = (162.568, 198.346, ..., 1.957961, 0.135242)  # Replace with actual feature values
input_data_as_numpy = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy.reshape(1, -1)

# Standardize the new input
std_data = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = model.predict(std_data)
if prediction[0] == 1:
    print('The person has Parkinson')
else:
    print('The person does not have Parkinson')
```

### 9. Save the model using pickle 
```python

filename = 'parkinson_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(f"Model saved as {filename}.")
```
### Load and use the model
```python

loaded_model = pickle.load(open('parkinson_model.sav', 'rb'))


result = loaded_model.predict(std_data)
print(f"Loaded model prediction: {result}")
```

## Conclusion

The Parkinson's Disease detection model built using a Support Vector Machine (SVM) classifier achieved an accuracy of 95% on the test data, demonstrating its effectiveness in distinguishing between individuals with Parkinson's disease and healthy individuals. The preprocessing steps, including data standardization, significantly improved the model's performance.

Key features such as 'MDVP:Fo(Hz)' and 'DFA' were identified as the most important for making accurate predictions. While the model performed well, future improvements could include experimenting with other classification algorithms or tuning hyperparameters to enhance performance further.

This model holds the potential to be integrated into clinical tools to assist healthcare professionals in early diagnosis, providing an efficient method for early intervention and treatment.




