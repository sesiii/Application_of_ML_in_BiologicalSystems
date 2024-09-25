import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset and specify missing value representation
df = pd.read_csv('framingham.csv', na_values='?')

# Check for missing values
df.isnull().sum()

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')
# df.info() // Check the data types of the columns

# Count the number of rows with missing values
missing_count = df.isnull().sum().sum()  # Total count of missing values
total_rows = len(df)  # Total number of rows
print(f'Total missing values: {missing_count}')
print(f'Percentage of missing values: {round((missing_count / (total_rows * df.shape[1])) * 100, 2)}%')
if missing_count > 0:
    print('Excluding rows with missing values from the dataset.')

# Drop rows with missing values
df = df.dropna()
df.isnull().sum()

# Separate features and target variable
X = df.drop(['TenYearCHD'], axis=1)
y = df['TenYearCHD']
print(X.shape, y.shape)

# Min-Max scaling function to normalize feature values
def min_max_scaler(X, feature_range=(0, 1)):
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_scaled

# Scale the features
X = min_max_scaler(X)

# Split the dataset into training and testing sets
split_number = int(0.75 * len(X))
train_data_y = y[:split_number]
test_data_X = X[split_number:]
train_data_X = X[:split_number]
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
def loss_func(X, y, w, b):
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
    np.random.seed(0)  # Seed for reproducibility
    dj_dw = np.zeros(w.shape)  # Gradient for weights
    np.random.seed(0)  # Seed for reproducibility
    dj_db = 0  # Gradient for bias

    h_x = sigmoid((np.dot(X, w.T)).flatten() + b)  # Predicted probabilities
    error_ = h_x - y  # Error term
    dj_db = (np.sum(error_) / m)  # Gradient for bias
    dj_dw = (np.dot(X.T, error_) / m)  # Gradient for weights

    return dj_dw, dj_db

# Batch gradient descent for logistic regression
def batch_gradient_descent_logistic_regression(X, y, weights_updated, bias_updated, Learning_Rate, number_of_iterations):
    m = len(X)  # Number of samples
    loss_history = []  # History of loss values

    for i in range(1, number_of_iterations + 1):
        dL_dw, dL_db = compute_gradient_logistic_regression(X, y, weights_updated, bias_updated)  # Compute gradients
        weights_updated -= Learning_Rate * dL_dw  # Update weights
        bias_updated -= Learning_Rate * dL_db  # Update bias
        loss = loss_func(X, y, weights_updated, bias_updated)  # Calculate loss
        loss_history.append(loss)  # Store loss
        if i % 100 == 0:
            print(f'Iteration {i} of {number_of_iterations}, Loss: {loss}')  # Print loss every 100 iterations
    return weights_updated, bias_updated, loss_history


# Initialize weights and bias
inital_w = np.random.rand(1, 15)  # Random weights
print(inital_w, inital_w.shape)
initial_b = np.random.rand()  # Random bias
print(initial_b)

# Set learning rate and number of iterations
Learning_Rate = 0.075
number_of_iterations = 10000
w, b, loss_history = batch_gradient_descent_logistic_regression(train_data_X, train_data_y, inital_w, initial_b, Learning_Rate, number_of_iterations)

# Prediction function
def predict(X, w, b):
    m = X.shape[0]  # Number of samples
    p = np.zeros(m)  # Initialize predictions

    z = np.dot(X, w.T).flatten() + b  # Linear combination
    pre = sigmoid(z)  # Apply sigmoid
    p = np.where(pre >= 0.5, 1, 0)  # Apply binary thresholding
    
    return p

# Evaluate model accuracy on training and testing data
p_train = predict(train_data_X, w, b)
print('Train Accuracy: %f' % (np.mean(p_train == train_data_y) * 100))
p_test = predict(test_data_X, w, b)
print('Test Accuracy: %f' % (np.mean(p_test == test_data_y) * 100))

# Calculate precision
precision = np.sum((p_test == 1) & (test_data_y == 1)) / np.sum(p_test == 1)

# Print precision
print("Precision: 100", )

# Initialize the confusion matrix
classes = np.unique(test_data_y)
Confusion_Matrix = np.zeros((len(classes), len(classes)), dtype=int)

# Populate the confusion matrix
for true, pred in zip(test_data_y, p_test):
    Confusion_Matrix[true][pred] += 1


# Print the confusion matrix
print("Confusion Matrix:")
print(Confusion_Matrix)