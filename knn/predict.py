import numpy as np
import csv
import sys

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def compute_ln_form_distance(test_elements_x,test_elements,n):
    sum_ = 0
    for i in range(len(test_elements_x)):
        sum_ = sum_ + abs(test_elements_x[i] - test_elements[i]) ** n

    sum_ = sum_ ** 1/n
    return sum_

def find_k_nearest_neighbours(train_X,test_elements,k,n):
    indices_dist_pairs = []
    index = 0
    for test_element_x in train_X:
        distance = compute_ln_form_distance(test_element_x,test_elements,n)
        indices_dist_pairs.append([index,distance])
        index += 1
    
    indices_dist_pairs.sort(key = lambda x : x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices


def classify_points_using_knn(train_X,train_Y,test_X,k,n):
    test_Y = []
    for test_elements in test_X:
        top_knn_indices = find_k_nearest_neighbours(train_X,test_elements,k,n)
        top_knn_labels = []

        for i in top_knn_indices:
            top_knn_labels.append(train_Y[i])

        Y_values = list(set(top_knn_labels))

        max_count = 0
        max_frequency_label = -1
        for y in Y_values:
            count = top_knn_labels.count(y)
            if(count > max_count):
                max_count = count
                max_frequency_label = y
        
        test_Y.append(max_frequency_label)
    
    return test_Y

def calculate_accuracy(predicted_Y,actual_Y):
    num_of_value_matched = 0
    total_no_of_observations = len(predicted_Y)
    for i in range(len(predicted_Y)):
        if predicted_Y[i]==actual_Y[i]:
            num_of_value_matched += 1
    
    return float(num_of_value_matched)/total_no_of_observations

def get_best_k_using_validation_set(train_X,train_Y,validation_split_percent,n):
    import math
    total_no_of_observations = len(train_X)
    train_length = math.floor(float(100 - validation_split_percent)/100 * total_no_of_observations)
    validation_X = train_X[train_length:]
    validation_Y = train_Y[train_length:]
    train_X = train_X[:train_length]
    train_Y = train_Y[:train_length]
    best_k = -1
    best_accuracy = 0
    for k in range(1,train_length+1):
        predicted_Y = classify_points_using_knn(train_X,train_Y,validation_X,k,n)
        accuracy = calculate_accuracy(predicted_Y,validation_Y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
        
    return best_k
    
def predict_target_values(test_X):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    train_X = np.genfromtxt("train_X_knn.csv",delimiter =',',dtype=np.float64,skip_header=1)
    train_Y = np.genfromtxt("train_Y_knn.csv",delimiter =',',dtype=np.float64)
    validation_split_percent = 30
    n = 2
    k = get_best_k_using_validation_set(train_X,train_Y,validation_split_percent,n)
    pred_Y = classify_points_using_knn(train_X,train_Y,test_X,k,n)
    return pred_Y
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file) #converting file object(csv_file) into csvwriter object(wr)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    pred_Y = np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 