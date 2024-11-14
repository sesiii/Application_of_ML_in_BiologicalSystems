import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('breastcancer.csv')

data = data.drop(columns=['Unnamed: 32'])

# Brief description of the dataset
print(data.head())
print(data.info())
print(data.describe())

# Target variable and feature columns
target = 'diagnosis'
features = data.columns.drop(['id', target])

#Data Manipulation
# Encode the target variable
data[target] = data[target].map({'M': 1, 'B': 0})

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#Model Building

# Train and evaluate SVM with different kernels
kernels = ['linear', 'poly', 'rbf']
svm_results = {}
for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    svm_results[kernel] = accuracy
    print(f'SVM with {kernel} kernel accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

# Train and evaluate Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest accuracy: {rf_accuracy}')
print(classification_report(y_test, y_pred_rf))


# Grid search for Neural Network
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), param_grid, n_jobs=-1, cv=3)
grid_search.fit(X_train_scaled, y_train)

print(f'Best parameters found: {grid_search.best_params_}')
best_nn = grid_search.best_estimator_

# Train and evaluate the best neural network
best_nn.fit(X_train_scaled, y_train)
y_pred_nn = best_nn.predict(X_test_scaled)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
print(f'Neural Network accuracy: {nn_accuracy}')
print(classification_report(y_test, y_pred_nn))


# Store results in a dictionary
results = {
    'SVM_linear': svm_results['linear'],
    'SVM_poly': svm_results['poly'],
    'SVM_rbf': svm_results['rbf'],
    'Random Forest': rf_accuracy,
    'Neural Network': nn_accuracy
}

# Compare the performance
print("Model Comparison:")
for model, result in results.items():
    print(f"{model}: {result}")

# Plot the results
import seaborn as sns
import matplotlib.pyplot as plt

model_names = list(results.keys())
accuracies = list(results.values())

sns.barplot(x=model_names, y=accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.show()

print("\n\n\n")


best_model = max(results, key=results.get)
print(f"The best-performing model is: {best_model}")
