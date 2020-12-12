import numpy as np
import csv

def get_train_data_for_class(train_X,train_Y,class_label):
    class_X = np.copy(train_X)
    class_Y = np.copy(train_Y)
    class_Y = np.where(class_Y == class_label,1,0)
    return class_X,class_Y

def import_data():
    X = np.genfromtxt("train_X_lg_v2.csv",dtype=np.float128,delimiter = ',',skip_header=1)
    Y = np.genfromtxt("train_Y_lg_v2.csv",dtype=np.float128,delimiter = ",")
    return X,Y

def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s

def compute_gradient_of_cost_function(X,Y,W,b):
    m = len(X)
    z = np.dot(X,W) + b 
    A = sigmoid(z)
    dz = A - Y
    dw = 1/m * np.dot(dz.T,X)
    db = 1/m * np.sum(dz)
    return dw.T,db

def compute_cost(X,Y,W,b):
    m = len(X)
    z = np.dot(X,W) + b
    A = sigmoid(z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = -1/m * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A))) 
    return cost

def optimize_weights_using_gradient_descent(X,Y,W,b,num_iterations,learning_rate):
    """iter_num = 0
    previous_itr_cost = 0
    while True:
        iter_num += 1
        dw,db = compute_gradient_of_cost_function(X,Y,W,b)
        W = W - (learning_rate * dw)
        b = b - (learning_rate * db)
        cost = compute_cost(X,Y,W,b)

        if iter_num%20000 ==0:
            print(iter_num,cost)

        if abs(cost - previous_itr_cost) < 0.0000001:
            print(iter_num,cost)
            break

        previous_itr_cost = cost"""

    for i in range(1,num_iterations+1):
        dw,db = compute_gradient_of_cost_function(X,Y,W,b)
        W = W - (learning_rate * dw)
        b = b - (learning_rate * db)
        cost = compute_cost(X,Y,W,b)
        if i%50000 == 0:
            print(i,cost)
    
    return W,b

def train_model(X,Y):
    
    
   
    learning_rate = [0.008,0.0079,0.00008,0.000001]
    
    set_of_w = []
    for i in range(4):
        X,Y = get_train_data_for_class(X,Y,i)
        Y = Y.reshape(len(X),1)
        W = np.zeros((X.shape[1],1))
        b = 0.0
        W,b = optimize_weights_using_gradient_descent(X,Y,W,b,200000,learning_rate[i])
        W = np.insert(W,0,b,axis = 0)
        set_of_w.append(W.T[0])
    
    set_of_w = np.array(set_of_w)
    return set_of_w

def save_model(w,weights_file_name):
    with open(weights_file_name,'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(w)
        weights_file.close()
 
if __name__ == "__main__":
    
    X,Y = import_data()
    W = train_model(X,Y)
    save_model(W,"WEIGHT_FILE.csv")
