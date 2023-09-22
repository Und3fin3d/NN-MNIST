import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

time_history = []
loss_history = []
accuracy_history = []

data = pd.read_csv('train.csv')
data = np.array(data)
n, i = data.shape
np.random.shuffle(data)

data_test = data[0:1000].T
test_labels = data_test[0]
test_images = data_test[1:]
test_images = test_images / 255

train = data[1000:n].T
i, n = train.shape
train_labels = train[0]
train_images = train[1:]
train_images = train_images / 255

def init_parameters():
    w1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(20, 10) * np.sqrt(2 / 10)
    b2 = np.zeros((20, 1))
    w3 = np.random.randn(10, 20) * np.sqrt(2 / 20)
    b3 = np.zeros((10, 1))
    w4 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b4 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3, w4,b4

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    z = np.exp(x) / sum(np.exp(x))
    return z
    
def forward_propagation(w1, b1, w2, b2, w3, b3, w4, b4, a0):
    z1 = np.matmul(w1,a0) + b1
    a1 = ReLU(z1)
    z2 = np.matmul(w2,a1) + b2
    a2 = ReLU(z2)
    z3 = np.matmul(w3,a2) + b3
    a3 = ReLU(z3)
    z4 = np.matmul(w4,a3) + b4
    a4 = softmax(z4)
    return z1, a1, z2, a2, z3, a3, z4, a4

def ReLU_deriv(x):
    return x > 0

def one_hot(label):
    y = np.zeros((label.size, label.max() + 1))
    y[np.arange(label.size), label] = 1
    y = y.T
    return y

def backward_propagation(z1, a1, z2, a2, z3, a3, a4, w2, w3, w4, a0, y):
    dC_dz4 = a4 - y
    dC_dw4 = 1 / n  * np.matmul(dC_dz4,a3.T)
    dC_db4 = 1 / n  * np.sum(dC_dz4)
    dC_dz3 = np.matmul(w4.T, dC_dz4) * ReLU_deriv(z3)
    dC_dw3 = 1 / n  * np.matmul(dC_dz3,a2.T)
    dC_db3 = 1 / n  * np.sum(dC_dz3)
    dC_dz2 = np.matmul(w3.T,dC_dz3) * ReLU_deriv(z2)
    dC_dw2 = 1 / n  *np.matmul(dC_dz2, a1.T)
    dC_db2 = 1 / n  * np.sum(dC_dz2)
    dC_dz1 = np.matmul(w2.T,dC_dz2) * ReLU_deriv(z1)
    dC_dw1 = 1 / n  * np.matmul(dC_dz1,a0.T)
    dC_db1 = 1 / n  * np.sum(dC_dz1)
    return dC_dw1, dC_db1, dC_dw2, dC_db2, dC_dw3, dC_db3, dC_dw4,dC_db4

def update_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dC_dw1, dC_db1, dC_dw2, dC_db2, dC_dw3, dC_db3, dC_dw4,dC_db4, alpha):
    w1 -= alpha * dC_dw1
    b1 -= alpha * dC_db1
    w2 -= alpha * dC_dw2
    b2 -= alpha * dC_db2
    w3 -= alpha * dC_dw3
    b3 -= alpha * dC_db3
    w4 -= alpha * dC_dw4
    b4 -= alpha * dC_db4
    return w1, b1, w2, b2, w3, b3, w4, b4

def predict(a4):
    return np.argmax(a4, 0)

def output(a0, w1, b1, w2, b2, w3, b3, w4, b4):
    _,_,_,_,_,_,_, a4 = forward_propagation(w1, b1, w2, b2, w3, b3, w4, b4, a0)
    return a4

def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size
def cross_entropy_loss(p, y):
    loss = -1/n * np.sum(np.sum(y * np.log(p)))
    
    return loss
def gradient_descent(a0, y, alpha, iterations):
    w1, b1, w2, b2, w3, b3, w4, b4 = init_parameters()
    start = time.perf_counter() 
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3, z4, a4 = forward_propagation(w1, b1, w2, b2, w3, b3, w4, b4, a0)
        dw1, db1, dw2, db2, dw3, db3, dw4, db4 = backward_propagation(z1, a1, z2, a2, z3, a3, a4, w2, w3, w4, a0, y)
        w1, b1, w2, b2, w3, b3, w4, b4 = update_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dw1, db1, dw2, db2, dw3, db3, dw4, db4, alpha)
        loss = cross_entropy_loss(a4, y)
        acc = accuracy(predict(a4), predict(y))
        loss_history.append(loss)
        accuracy_history.append(acc)
        time_history.append(time.perf_counter()-start)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Loss:",loss)
            print("Accuracy:", acc)
    return w1, b1, w2, b2, w3, b3, w4, b4



w1, b1, w2, b2, w3, b3, w4, b4 = gradient_descent(train_images, one_hot(train_labels), 0.1,2000)    
plt.plot(loss_history)
plt.title('Loss during training, 3 hidden layer')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss3h.png')
plt.clf()
plt.plot(accuracy_history)
plt.title('Accuracy during training, 3 hidden layer')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('Accuracy3h.png')
plt.clf()
plt.plot(time_history,loss_history)
plt.title('Loss during training, 3 hidden layer')
plt.xlabel('Seconds')
plt.ylabel('Loss')
plt.savefig('Losstime3h.png')
plt.clf()
plt.plot(time_history,accuracy_history)
plt.title('Accuracy during training, 3 hidden layer')
plt.xlabel('Seconds')
plt.ylabel('Accuracy')
plt.savefig('Accuracytime3h.png')


unseen = predict(output(test_images,w1, b1, w2, b2, w3, b3, w4, b4))
with open("Results.txt","w") as f:
    f.write("3 Hidden " + str(accuracy(unseen, test_labels)))