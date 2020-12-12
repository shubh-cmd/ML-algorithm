import numpy as np
import csv
import sys
from train import sigmoid
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_lg.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights


def predict_target_values(test_X, weights):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    
    b = weights[0]
    #w = np.array(weights)
    #weights = w.T
    weights = weights.reshape(len(weights),1)
    w = np.delete(weights,0,axis = 0)
    z = np.dot(test_X,w) + b
    pred_Y = sigmoid(z)
    #pred_Y = np.where(a >= 0.5,1,0)
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHT_FILE.csv")
    Y = []
    for i in range(4):
        pred_Y = predict_target_values(test_X, weights[i])
        Y.append(pred_Y.T[0])
    
    pred_Y = []
    
    Y = np.array(Y)
    for i in range(len(Y[0])):
        j = Y[:,i]
        index = np.where(j == np.amax(j))
        g = list(index[0])
        pred_Y.append(g)

    pred_Y = np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 
