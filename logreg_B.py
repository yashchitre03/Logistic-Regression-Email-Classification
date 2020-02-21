
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math

def predict(X, w):
    n_ts = X.shape[0]
    # use w for prediction
    pred = np.zeros(n_ts)       # initialize prediction vector
    for i in range(n_ts):
        # TODO
        pred[i] = 1 if (np.dot(w.T, X[i]) > 0) else 0   # compute your prediction
    
    return pred

def accuracy(X, y, w):
    correct, acc = 0, 0
    y_pred = predict(X, w)
    # TODO
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            correct += 1                # if prediction is correct, it increments the variable
    acc = correct / len(y)
    return acc                          # returns the accuracy

def logistic_reg(X_tr, y_tr, X_ts, y_ts, lr, k_t):
    lda = 2**k_t
    #perform gradient descent
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1000         # maximum iteration

    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        for i in range(n_tr):
            x = np.dot(w, X_tr[i])          # calculates the dot product
            temp = (1 / (1 + np.exp(-x))) if (x >= 0) else (np.exp(x) / (1 + np.exp(x)))        # uses sigmoid function depending upon the value of x

            for j in range(n_vars):
                # TODO
                grad[j] = grad[j] + ((y_tr[i] - temp) * X_tr[i][j])         # updates the gradient

        for j in range(n_vars):
            grad[j] = grad[j] - (lda*w[j])          # subtract lambda*w for each j variable
            
        # TODO
        w_new = w + lr * grad
        
        if math.isnan(np.mean(grad)):       # if we get values too small or large to handle, we break out of the loop
            break

        if iter%50 == 0:
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))

        # stopping criteria and perform update if not stopping
        if (np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new

        if (iter >= max_iter):
            break

    test_accuracy  = accuracy(X_ts, y_ts, w)
    train_accuracy = accuracy(X_tr, y_tr, w)
    return test_accuracy, train_accuracy


# read files
D_tr = genfromtxt('spambasetrain.csv', delimiter = ',', encoding = 'utf-8')
D_ts = genfromtxt('spambasetest.csv', delimiter = ',', encoding = 'utf-8')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
n_ts = D_ts.shape[0]

# add 1 as feature
X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

# set learning rate
k = [-8, -6, -4, -2, 0, 2]
lr = 1e-3
final_test = []
final_train = []

for each in k:
    test_accuracy, train_accuracy = logistic_reg(X_tr, y_tr, X_ts, y_ts, lr, each)
    final_test.append(test_accuracy)
    final_train.append(train_accuracy)
    print('train accuracy = {0}, test accuracy = {1}'.format(str(train_accuracy), str(test_accuracy)))
# plot for different values of k
plt.title('K vs Accuracy')
plt.plot(k, final_train, label = 'Training Accuracy')
plt.plot(k, final_test, label = 'Testing Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()