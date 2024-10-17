import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.error = y - output
        self.delta_output = self.error * self.sigmoid_derivative(output)
        
        self.error_hidden = np.dot(self.delta_output, self.w2.T)
        self.delta_hidden = self.error_hidden * self.sigmoid_derivative(self.a1)
        
        self.w2 += self.learning_rate * np.dot(self.a1.T, self.delta_output)
        self.b2 += self.learning_rate * np.sum(self.delta_output, axis=0, keepdims=True)
        self.w1 += self.learning_rate * np.dot(X.T, self.delta_hidden)
        self.b1 += self.learning_rate * np.sum(self.delta_hidden, axis=0, keepdims=True)
    
    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def predict(self, X):
        return self.forward(X)

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('MEDV', axis=1).values
    y = data['MEDV'].values.reshape(-1, 1)
    
    # Normalize X using Min-Max scaling
    scaler_X = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)
    
    # Normalize y using Min-Max scaling
    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y)
    
    return X_normalized, y_normalized, scaler_y

def evaluate_model(model, X, y, scaler_y):
    predictions = model.predict(X)
    predictions_original = scaler_y.inverse_transform(predictions)
    y_original = scaler_y.inverse_transform(y)
    mse = np.mean((predictions_original - y_original) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def cross_validate(X, y, input_size, hidden_size, output_size, learning_rate, epochs, k_folds):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
        model.train(X_train, y_train, epochs)
        
        rmse = evaluate_model(model, X_test, y_test, scaler_y)
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores), np.std(rmse_scores)

if __name__ == "__main__":
    # Load and preprocess data
    X, y, scaler_y = load_and_preprocess_data('housing.csv')
    input_size = X.shape[1]
    output_size = 1
    epochs = 1000

    # Configurations to test
    configs = [
        {"hidden_size": 3, "learning_rate": 0.01},
        {"hidden_size": 4, "learning_rate": 0.001},
        {"hidden_size": 5, "learning_rate": 0.0001}
    ]

    for config in configs:
        print(f"\nConfiguration: Hidden neurons = {config['hidden_size']}, Learning rate = {config['learning_rate']}")
        
        # 5-fold cross-validation
        mean_rmse_5, std_rmse_5 = cross_validate(X, y, input_size, config['hidden_size'], output_size, config['learning_rate'], epochs, 5)
        print(f"5-fold CV - Mean RMSE: {mean_rmse_5:.4f} (±{std_rmse_5:.4f})")
        
        # 10-fold cross-validation
        mean_rmse_10, std_rmse_10 = cross_validate(X, y, input_size, config['hidden_size'], output_size, config['learning_rate'], epochs, 10)
        print(f"10-fold CV - Mean RMSE: {mean_rmse_10:.4f} (±{std_rmse_10:.4f})")