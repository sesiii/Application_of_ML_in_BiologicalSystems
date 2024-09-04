import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def preprocess_data(df):
    # Convert categorical variables to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Separate features and target variable
    X = df.drop('charges', axis=1).values
    y = df['charges'].values

    # Convert X to float64 BEFORE feature scaling
    X = X.astype(np.float64)

    # Feature Scaling (Standardization)
    for i in range(X.shape[1]):  # Iterate through columns
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

    return X, y

# Evaluate Model
def evaluate_model(y_true, y_predicted):
    mse = np.mean((y_true - y_predicted) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_predicted) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, rmse, r2

# Load the dataset
df = pd.read_csv('data_insurance.csv')

# Preprocess the data
X, y = preprocess_data(df)

# Split the dataset into training and test sets (80% train, 20% test)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predicted = model.predict(X_test)

# Evaluate the model
mse, rmse, r2 = evaluate_model(y_test, y_predicted)

# Print the results
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)


# ----------------------------------------
# Plotting 
# ----------------------------------------

# 1. Predicted vs. Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predicted, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs. Actual Charges")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.show()

# 2. Residuals Plot
residuals = y_test - y_predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_predicted, residuals, alpha=0.5)
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at 0
plt.show()

