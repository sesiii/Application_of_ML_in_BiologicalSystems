# Application of Machine Learning in Biological Systems

## Project 2: Logistic Regression for Heart Disease Prediction

### Overview
This project implements a logistic regression model to predict the likelihood of heart disease based on a dataset from the Framingham Heart Study. The model processes the data, handles missing values, scales features, and evaluates performance using accuracy and a confusion matrix.

### Dependencies
Make sure to install the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

You can install these using pip:
```bash
pip install numpy pandas matplotlib seaborn
```

### Dataset
The dataset used in this project is `framingham.csv`, which contains various health metrics and a target variable indicating the presence of heart disease over a ten-year period.

### Steps
1. **Load the Dataset**: The dataset is loaded, and missing values are represented as `NaN`.
2. **Data Preprocessing**:
    - Check for and count missing values.
    - Convert all columns to numeric types.
    - Drop rows with missing values.
3. **Feature and Target Separation**: Features are separated from the target variable (`TenYearCHD`).
4. **Feature Scaling**: Min-Max scaling is applied to normalize feature values.
5. **Data Splitting**: The dataset is split into training and testing sets.
6. **Model Training**:
    - Implement logistic regression using batch gradient descent.
    - Initialize weights and bias randomly.
    - Train the model over a specified number of iterations.
7. **Prediction**: The model predicts outcomes for both training and testing datasets.
8. **Evaluation**:
    - Calculate and print accuracy for both training and testing datasets.
    - Generate and display a confusion matrix.

### Functions
- `min_max_scaler(X)`: Normalizes feature values to a specified range.
- `sigmoid(z)`: Applies the sigmoid function to map values between 0 and 1.
- `loss_function(X, y, w, b)`: Computes the cross-entropy loss.
- `compute_gradient_logistic_regression(X, y, w, b)`: Calculates gradients for weights and bias.
- `batch_gradient_descent_logistic_regression(X, y, w_in, b_in, alph_val, num_iters)`: Performs batch gradient descent to optimize weights and bias.
- `predict(X, w, b)`: Generates predictions based on the trained model.

### Results
The model's performance is evaluated using accuracy metrics and a confusion matrix, providing insights into its predictive capabilities.

### Conclusion
This project demonstrates the application of logistic regression in predicting heart disease, showcasing the importance of data preprocessing, model training, and evaluation in machine learning.

### License
This project is licensed under the MIT License.  