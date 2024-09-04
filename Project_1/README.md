# Project 1: Application of Machine Learning in Biological Systems

This project aims to build a linear regression model to predict medical insurance charges based on various features. The dataset used for this project is `data_insurance.csv`.

## Data Description
The dataset contains the following columns:
- `age`: age of the individual
- `sex`: gender of the individual (male or female)
- `bmi`: body mass index (ratio of weight to height)
- `children`: number of children the individual has
- `smoker`: whether the individual is a smoker or not (yes or no)
- `region`: region of the individual (northeast, northwest, southeast, southwest)
- `charges`: medical insurance charges

## Approach
1. Preprocessing the data:
    - Converting categorical variables to numerical using one-hot encoding.
    - Separating features and the target variable.
    - Converting features to float64 and performing feature scaling (standardization).

2. Splitting the dataset into training and test sets:
    - Using an 80% train and 20% test split ratio.

3. Building and training the linear regression model:
    - Initializing the model with a learning rate of 0.01 and 1000 iterations.
    - Updating the model weights and bias using gradient descent.

4. Making predictions on the test set:
    - Using the trained model to predict medical insurance charges.

5. Evaluating the model:
    - Calculating Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) to assess the model's performance.

## Results
The model's performance on the test set is as follows:
- Mean Squared Error (MSE): [mse] : 37164301.71146567
- Root Mean Squared Error (RMSE): [rmse] : 6096.253087878297
- R-squared (R²): [r2] : 0.7585604010433598

## Insights
- The "Predicted vs. Actual Values" plot shows the relationship between the actual charges and the predicted charges. The red dashed line represents the ideal scenario where the predicted charges perfectly match the actual charges.
- The "Residuals Plot" shows the difference between the predicted charges and the actual charges. The red dashed line represents the zero residual line, indicating the ideal scenario where the residuals are centered around zero.

Please refer to the code and the generated plots for a more detailed analysis.
