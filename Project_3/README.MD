# Decision Tree Classifier for Breast Cancer Survival Prediction

## Overview

This project implements a decision tree classifier from scratch to predict the survival status ("Alive" or "Dead") of breast cancer patients based on various clinical features. The decision tree is built using the Gini index as the splitting criterion. 

## Dataset

The project uses the "Breast_Cancer.csv" dataset, which contains information about breast cancer patients, including features like age, race, marital status, tumor stage, tumor size, and estrogen status. The target variable is "Status", indicating whether the patient is alive or dead.

## Approach

1. **Data Loading and Preprocessing:**
   - The dataset is loaded using Pandas.
   - Missing values (if any) are handled (e.g., by removing rows with missing data).
   - The "Status" column (target variable) is encoded: "Alive" is mapped to 1, and "Dead" is mapped to 0. This allows the decision tree to work with numerical data.

2. **Data Splitting:**
   - The dataset is split into training and testing sets using `train_test_split` from scikit-learn. A typical split ratio is 80% for training and 20% for testing.

3. **Decision Tree Implementation:**
   - A `DecisionTreeClassifier` class is implemented from scratch.
   - The class includes methods for calculating Gini impurity, Gini gain, building the decision tree recursively (`__decision_tree`), fitting the tree to the training data (`fit`), making predictions (`predict`), and evaluating accuracy (`score`).

4. **Tree Building:**
   - The decision tree is built using the Gini index as the splitting criterion.
   - The tree is built recursively:
     - At each node, the feature that maximizes the Gini gain is selected as the splitting feature.
     - The dataset is then split into subsets based on the unique values of the selected feature.
     - The process is repeated recursively for each subset until a stopping condition is met (e.g., all samples in a node have the same class label, a maximum depth is reached, or no more features are available for splitting).

5. **Model Fitting and Evaluation:**
   - The `DecisionTreeClassifier` is instantiated.
   - The `fit` method is called to build the decision tree using the training data and the specified `metric` (Gini index) and `max_depth`.
   - The `predict` method is used to make predictions on the test data.
   - The `score` method is used to calculate the accuracy of the model on the test data.

6. **Epoch-based Evaluation (for demonstration):**
   - Although decision trees are not typically trained iteratively like neural networks, an epoch-based loop is included for demonstration purposes. 
   - In each epoch (iteration), the accuracy and loss (1 - accuracy) are calculated for both the training and validation sets.
   - Note that the decision tree structure remains the same across epochs.

7. **Plotting Accuracy and Loss:**
   - The training and validation accuracy and loss are plotted over the epochs using Matplotlib.
   - The plots will likely show flat lines, indicating that the tree's performance does not change with epochs (as expected for decision trees).

## Results

The following results were obtained using a decision tree with a `max_depth` of 2:

- **Training Accuracy:** 0.9674
- **Training Loss:** 0.0326
- **Validation Accuracy:** 0.8646
- **Validation Loss:** 0.1354

The model achieved an accuracy of **86.46%** on the test data. The accuracy and loss plots (shown in the code output) demonstrate the stability of the tree's performance over multiple evaluations.

## Conclusion

This project demonstrates the implementation of a decision tree classifier from scratch for breast cancer survival prediction. The Gini index is used as the splitting criterion, and the tree's performance is evaluated on a test set. The project also includes an epoch-based evaluation loop and plots to illustrate the stability of the decision tree's performance.

## Future Work

- Explore other splitting criteria (e.g., information gain).
- Implement pruning techniques to reduce overfitting.
- Experiment with different hyperparameters (e.g., maximum depth) to optimize performance.
- Compare the performance of the implemented decision tree with other classification algorithms.