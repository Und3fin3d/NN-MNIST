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
    w1 = np.random.randn(20, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((20, 1))
    w2 = np.random.randn(10, 20) * np.sqrt(2 / 20)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    z = np.exp(x) / sum(np.exp(x))
    return z
    
def forward_propagation(w1, b1, w2, b2, a0):
    z1 = np.matmul(w1,a0) + b1
    a1 = ReLU(z1)
    z2 = np.matmul(w2,a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def ReLU_deriv(x):
    return x > 0

def one_hot(label):
    y = np.zeros((label.size, label.max() + 1))
    y[np.arange(label.size), label] = 1
    y = y.T
    return y

def backward_propagation(z1, a1, z2, a2, w2, a0, y):
    dC_dz2 = a2 - y
    dC_dw2 = 1 / n  * np.matmul(dC_dz2,a1.T)
    dC_db2 = 1 / n  * np.sum(dC_dz2)
    dC_dz1 = np.matmul(w2.T,dC_dz2) * ReLU_deriv(z1)
    dC_dw1 = 1 / n  *  np.matmul(dC_dz1,a0.T)
    dC_db1 = 1 / n  * np.sum(dC_dz1)
    return dC_dw1, dC_db1, dC_dw2, dC_db2

def update_parameters(w1, b1, w2, b2, dC_dw1, dC_db1, dC_dw2, dC_db2, alpha):
    w1 -= alpha * dC_dw1
    b1 -= alpha * dC_db1
    w2 -= alpha * dC_dw2
    b2 -= alpha * dC_db2
    return w1, b1, w2, b2

def predict(a2):
    return np.argmax(a2, 0)

def output(a0, w1, b1, w2, b2):
    _,_,_,a2 = forward_propagation(w1, b1, w2, b2,a0)
    return a2

def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size
def cross_entropy_loss(p, y):
    loss = -1/n * np.sum(np.sum(y * np.log(p)))
    
    return loss
def gradient_descent(a0, y, alpha, iterations):
    w1, b1, w2, b2 = init_parameters()
    start = time.perf_counter() 
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, a0)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, a0, y)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        loss = cross_entropy_loss(a2, y)
        acc = accuracy(predict(a2), predict(y))
        loss_history.append(loss)
        accuracy_history.append(acc)
        time_history.append(time.perf_counter()-start)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Loss:",loss)
            print("Accuracy:", acc)
    return w1, b1, w2, b2



w1, b1, w2, b2 = gradient_descent(train_images, one_hot(train_labels), 0.1,2000)    
plt.plot(loss_history)
plt.title('Loss during training, 1 hidden layer')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss1h.png')
plt.clf()
plt.plot(accuracy_history)
plt.title('Accuracy during training, 1 hidden layer')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('Accuracy1h.png')
plt.clf()
plt.plot(time_history,loss_history)
plt.title('Loss during training, 1 hidden layer')
plt.xlabel('Seconds')
plt.ylabel('Loss')
plt.savefig('Losstime1h.png')
plt.clf()
plt.plot(time_history,accuracy_history)
plt.title('Accuracy during training, 1 hidden layer')
plt.xlabel('Seconds')
plt.ylabel('Accuracy')
plt.savefig('Accuracytime1h.png')


unseen = predict(output(test_images,w1, b1, w2, b2))
with open("Results.txt","w") as f:
    f.write("1 Hidden " + str(accuracy(unseen, test_labels)))