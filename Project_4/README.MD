# Backpropagation Neural Network Implementation

This project implements a flexible backpropagation neural network using NumPy and applies it to the Boston Housing dataset.

## Requirements

- Python 3.7+
- NumPy
- pandas
- scikit-learn

## How to Run

1. Ensure you have all the required libraries installed:
   ```
   pip install numpy pandas scikit-learn
   ```

2. Place the `housing.csv` file in the same directory as the script.

3. Run the script:
   ```
   python neural_network.py
   ```

The script will automatically run the neural network with the specified configurations and display the results for both 5-fold and 10-fold cross-validation.

## Configuration

The script tests three configurations:
a) Number of neurons in hidden layer = 3, learning rate = 0.01
b) Number of neurons in hidden layer = 4, learning rate = 0.001
c) Number of neurons in hidden layer = 5, learning rate = 0.0001

Each configuration is tested with both 5-fold and 10-fold cross-validation.

## Output

The script will output the mean RMSE and standard deviation for each configuration and cross-validation fold.