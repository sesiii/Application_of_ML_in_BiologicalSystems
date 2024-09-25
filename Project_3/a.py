import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Breast_Cancer.csv')

# Preprocess the data (handle missing values if any)
data = data.dropna()

# Encode 'Status' column: Alive -> 1, Dead -> 0
data['Status'] = data['Status'].map({'Alive': 1, 'Dead': 0})

# Separate features and target
X = data.drop('Status', axis=1).values
Y = data['Status'].values

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Implement the decision tree using the provided DecisionTreeClassifier class
class TreeNode:
    def __init__(self, data, output):
        self.data = data
        self.children = {}
        self.output = output  # Stores 'Alive' or 'Dead'
        self.index = -1
        
    def add_child(self, feature_value, obj):
        self.children[feature_value] = obj

class DecisionTreeClassifier:
    def __init__(self):
        self.__root = None

    def __count_unique(self, Y):
        d = {}
        for i in Y:
            if i not in d:
                d[i] = 1
            else:
                d[i] += 1
        return d

    def __entropy(self, Y):
        freq_map = self.__count_unique(Y)
        entropy_ = 0
        total = len(Y)
        for i in freq_map:
            p = freq_map[i] / total
            entropy_ += (-p) * math.log2(p)
        return entropy_

    def __gain_ratio(self, X, Y, selected_feature):
        info_orig = self.__entropy(Y)
        info_f = 0
        split_info = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        initial_size = df.shape[0]
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            info_f += (current_size / initial_size) * self.__entropy(df1[df1.shape[1] - 1])
            split_info += (-current_size / initial_size) * math.log2(current_size / initial_size)
        if split_info == 0:
            return math.inf
        info_gain = info_orig - info_f
        gain_ratio = info_gain / split_info
        return gain_ratio

    def __gini_index(self, Y):
        freq_map = self.__count_unique(Y)
        gini_index_ = 1
        total = len(Y)
        for i in freq_map:
            p = freq_map[i] / total
            gini_index_ -= p ** 2
        return gini_index_

    def __gini_gain(self, X, Y, selected_feature):
        gini_orig = self.__gini_index(Y)
        gini_split_f = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        initial_size = df.shape[0]
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            gini_split_f += (current_size / initial_size) * self.__gini_index(df1[df1.shape[1] - 1])
        gini_gain_ = gini_orig - gini_split_f
        return gini_gain_

    def __decision_tree(self, X, Y, features, level, metric, classes, max_depth):
        if len(set(Y)) == 1 or len(features) == 0 or level == max_depth: 
            # Base cases (leaf node conditions)
            print("Level", level)
            freq_map = self.__count_unique(Y)
            output = None
            max_count = -math.inf
            for i in classes:
                if i not in freq_map:
                    print("Count of", i, "=", 0)
                else:
                    if freq_map[i] > max_count:
                        output = i  # Assign the 'Status' string
                        max_count = freq_map[i]
                    print("Count of", i, "=", freq_map[i])
            if metric == "gain_ratio":
                print("Current Entropy  is =", self.__entropy(Y))
            elif metric == "gini_index":
                print("Current Gini Index is =", self.__gini_index(Y))
            print("Reached leaf Node")
            print()
            return TreeNode(None, output)
        
        max_gain = -math.inf
        final_feature = None
        for f in features:
            if metric == "gain_ratio":
                current_gain = self.__gain_ratio(X, Y, f)
            elif metric == "gini_index":
                current_gain = self.__gini_gain(X, Y, f)
            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = f
        print("Level", level)
        freq_map = self.__count_unique(Y)
        output = None
        max_count = -math.inf
        for i in classes:
            if i not in freq_map:
                print("Count of", i, "=", 0)
            else:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]
                print("Count of", i, "=", freq_map[i])
        if metric == "gain_ratio":
            print("Current Entropy is =", self.__entropy(Y))
            print("Splitting on feature  X[", final_feature, "] with gain ratio ", max_gain, sep="")
            print()
        elif metric == "gini_index":
            print("Current Gini Index is =", self.__gini_index(Y))
            print("Splitting on feature  X[", final_feature, "] with gini gain ", max_gain, sep="")
            print()
        unique_values = set(X[:, final_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        current_node = TreeNode(final_feature, output)
        index = features.index(final_feature)
        features.remove(final_feature)
        for i in unique_values:
            df1 = df[df[final_feature] == i]
            node = self.__decision_tree(df1.iloc[:, 0:df1.shape[1] - 1].values, df1.iloc[:, df1.shape[1] - 1].values, features, level + 1, metric, classes, max_depth)
            current_node.add_child(i, node)
        features.insert(index, final_feature)
        return current_node

    def fit(self, X, Y, metric="gain_ratio", max_depth=2): # Set max_depth here
        features = [i for i in range(len(X[0]))]
        classes = set(Y)
        level = 0
        if metric != "gain_ratio":
            if metric != "gini_index":
                metric = "gain_ratio"
        self.__root = self.__decision_tree(X, Y, features, level, metric, classes, max_depth)

    def __predict_for(self, data, node):
        if len(node.children) == 0:
            return node.output  # Return the 'Status' string
        val = data[node.data]
        if val not in node.children:
            return node.output
        return self.__predict_for(data, node.children[val])

    def predict(self, X):
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.__predict_for(X[i], self.__root)
        return Y

    def score(self, X, Y):
        Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                count += 1
        return count / len(Y_pred)


# Fit the model using the Gini index criterion and max_depth=2
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train, metric='gini_index', max_depth=2)  # Limit to two levels
# clf = DecisionTreeClassifier()

# Lists to store accuracy and loss for each epoch
train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

# Iterate through epochs (iterations)
for epoch in range(1, 21):
    print(f"Epoch {epoch}:")
    clf.fit(X_train, Y_train, metric='gini_index')  # Fit the tree

    # Calculate training accuracy and loss
    Y_train_pred = clf.predict(X_train)
    train_accuracy = clf.score(X_train, Y_train)
    train_loss = 1 - train_accuracy  # Using (1 - accuracy) as a proxy for loss
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)

    # Calculate validation accuracy and loss
    Y_val_pred = clf.predict(X_test)
    val_accuracy = clf.score(X_test, Y_test)
    val_loss = 1 - val_accuracy 
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)

    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Training Loss: {train_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print("-" * 20)

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 21), train_accuracies, label='Training Accuracy')
plt.plot(range(1, 21), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 21), train_losses, label='Training Loss')
plt.plot(range(1, 21), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()
# Predict and evaluate the model
Y_pred = clf.predict(X_test)
print("Accuracy:", clf.score(X_test, Y_test))  # Print accuracy only