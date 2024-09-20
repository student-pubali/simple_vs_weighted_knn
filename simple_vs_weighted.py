import numpy as np
import pandas as pd
from collections import Counter
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math
class kNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        print("Training Done")

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distance = []
            for i, train_point in enumerate(self.X_train):
                dist = ((test_point[0] - train_point[0])**2 + (test_point[1] - train_point[1])**2)**0.5
                distance.append((i, dist))
            distance.sort(key=operator.itemgetter(1))
            predictions.append(self.weighted_classify(distance[:self.k]))
        return predictions
    def classify(self, distance):
        labels = [self.Y_train[i[0]] for i in distance]
        return Counter(labels).most_common(1)[0][0]
    def weighted_classify(self, distances):
        # Calculate the weighted vote for each class based on the distance
        weights = {}
        for i, dist in distances:
            label = self.Y_train[i]
            weight = 1 / (dist + 1e-5)  # Avoid division by zero (add a small epsilon)
            if label in weights:
                weights[label] += weight
            else:
                weights[label] = weight
        # Return the label with the highest weighted sum
        return max(weights, key=weights.get)
#Load the dataset
data = pd.read_csv('Social_Network_Ads.csv')
data.head()    
# Split data into features and labels
X = data.iloc[:, 2:4].values
Y = data.iloc[:, -1].values
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 
# List of k values to test
n_samples = int(math.sqrt(len(X_train)))

# Initialize lists to store accuracies
simple_knn_accuracies = []
weighted_knn_accuracies = []

# Loop over different values of k
for i in range(1,n_samples+1):
    simple_knn = kNearestNeighbors(k=i)
    simple_knn.fit(X_train,Y_train)
    simple_predictions = simple_knn.predict(X_test)
    simple_acc = accuracy_score(Y_test,simple_predictions)
    simple_knn_accuracies.append(simple_acc)
    #Weighted knn
    weighted_knn = kNearestNeighbors(k=i)
    weighted_knn.fit(X_train,Y_train)
    weighted_predictions = weighted_knn.predict(X_test)
    weighted_acc = accuracy_score(Y_test,weighted_predictions)
    weighted_knn_accuracies.append(weighted_acc)

plt.plot(range(1,n_samples+1),simple_knn_accuracies,label='Simple KNN',marker='o') 
plt.plot(range(1,n_samples+1),weighted_knn_accuracies,label='Weighted KNN',marker='x') 
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title("Simple knn vs Weighted knn")
plt.show() 
simple_optimal_k = simple_knn_accuracies.index(max(simple_knn_accuracies))
weighted_optimal_k = weighted_knn_accuracies.index(max(weighted_knn_accuracies))
knn = kNearestNeighbors(k=simple_optimal_k)
knn.fit(X_train,Y_train)
def predict_new():
    age = int(input("Enter the age: "))
    salary = int(input("Enter the salary: "))

    # Construct the new input
    X_new = np.array([age, salary]).reshape(1, -1)

    # Standardize the new input
    X_new = scaler.transform(X_new)
 
    # Predict the result
    result = knn.predict(X_new)[0]

    # Output the prediction
    if result == 0:
        print("Will not purchase")
    else:
        print("Will purchase")

# Run the prediction function for a new input
predict_new()

   