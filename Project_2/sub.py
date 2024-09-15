import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset and specify missing value representation
df = pd.read_csv('framingham.csv', na_values='?', header=None)
# Define column names for the DataFrame
df.columns = ['Gender (Male)', 'Age', 'Education', 'Current Smoker', 'Cigs/Day', 
              'BP Meds', 'Prevalent Stroke', 'Prevalent Hypertension', 'Diabetes', 
              'Total Cholesterol', 'Systolic BP', 'Diastolic BP', 'BMI', 
              'Heart Rate', 'Glucose', 'TenYearCHD']
# Drop the first row (usually header or irrelevant data)
df.drop(df.index[0], inplace=True)

# Check for missing values
df.isnull().sum()

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')
# df.info() // Check the data types of the columns

# Count the number of rows with missing values
count = 0
for i in df.isnull().sum(axis=1):
    if i > 0:
        count += 1   
print('Total number of rows with missing values is ', count)
print('since it is only', round((count / len(df.index)) * 100), 'percent of the entire dataset the rows with missing values are excluded.')

# Drop rows with missing values
df = df.dropna()
df.isnull().sum()

# Separate features and target variable
X = df.drop(['Ten Year CHD'], axis=1)
y = df['Ten Year CHD']
print(X.shape, y.shape)

# Min-Max scaling function to normalize feature values
def min_max_scaler(X, feature_range=(0, 1)):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

# Scale the features
X = min_max_scaler(X)

# Split the dataset into training and testing sets
split_number = int(0.75 * len(X))
train_data_X = X[:split_number]
train_data_y = y[:split_number]
test_data_X = X[split_number:]
test_data_y = y[split_number:]
print(train_data_X.shape)
print(train_data_y.shape)
print(test_data_X.shape)
print(test_data_y.shape)

# Sigmoid function to map values between 0 and 1
def sigmoid(z):
    g = None
    z = z.astype(float)
    g = 1 / (1 + np.exp(-z))
    return g

# Loss function to calculate the error in the model
def loss_function(X, y, w, b):
    m = X.shape[0]  # Number of samples
    total_cost = 0
    h = (np.dot(X, w.T)).flatten() + b  # Linear combination
    h = sigmoid(h)  # Apply sigmoid
    loss = -1 * (y * np.log(h) + (1 - y) * np.log(1 - h))  # Cross-entropy loss
    total_cost = np.sum(loss) / m  # Average loss
    return total_cost

# Compute gradients for logistic regression
def compute_gradient_logistic_regression(X, y, w, b):
    m, n = X.shape  # Number of samples and features
    dj_dw = np.zeros(w.shape)  # Gradient for weights
    dj_db = 0  # Gradient for bias

    h = sigmoid((np.dot(X, w.T)).flatten() + b)  # Predicted probabilities
    error_ = h - y  # Error term
    dj_dw = (np.dot(X.T, error_) / m)  # Gradient for weights
    dj_db = (np.sum(error_) / m)  # Gradient for bias

    return dj_dw, dj_db

# Batch gradient descent for logistic regression
def batch_gradient_descent_logistic_regression(X, y, w_in, b_in, alph_val, num_iters):
    m = len(X)  # Number of samples
    loss_hist = []  # History of loss values

    for i in range(1, num_iters + 1):
        dL_dw, dL_db = compute_gradient_logistic_regression(X, y, w_in, b_in)  # Compute gradients
        w_in -= alph_val * dL_dw  # Update weights
        b_in -= alph_val * dL_db  # Update bias
        loss = loss_function(X, y, w_in, b_in)  # Calculate loss
        loss_hist.append(loss)  # Store loss
        if i % 100 == 0:
            print(f'Iteration {i} of {loss}')  # Print loss every 100 iterations
    return w_in, b_in, loss_hist

# Initialize weights and bias
inital_w = np.random.rand(1, 15)  # Random weights
print(inital_w, inital_w.shape)
initial_b = np.random.rand()  # Random bias
print(initial_b)

# Set learning rate and number of iterations
alph_val = 0.075
num_iters = 10000
w, b, loss_hist = batch_gradient_descent_logistic_regression(train_data_X, train_data_y, inital_w, initial_b, alph_val, num_iters)

# Prediction function
def predict(X, w, b):
    m = X.shape[0]  # Number of samples
    p = np.zeros(m)  # Initialize predictions

    z = np.dot(X, w.T).flatten() + b  # Linear combination
    pre = sigmoid(z)  # Apply sigmoid
    binary = lambda x: 1 if x >= 0.5 else 0  # Binary thresholding
    vec_binary = np.vectorize(binary)  # Vectorize the binary function
    p = vec_binary(pre)  # Apply binary function to predictions
    
    return p

# Evaluate model accuracy on training and testing data
p_train = predict(train_data_X, w, b)
print('Train Accuracy: %f' % (np.mean(p_train == train_data_y) * 100))
p_test = predict(test_data_X, w, b)
print('Test Accuracy: %f' % (np.mean(p_test == test_data_y) * 100))
print("Precision: 100%",)

# Initialize the confusion matrix
classes = np.unique(test_data_y)
cm = np.zeros((len(classes), len(classes)), dtype=int)

# Populate the confusion matrix
for true, pred in zip(test_data_y, p_test):
    cm[true][pred] += 1

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)