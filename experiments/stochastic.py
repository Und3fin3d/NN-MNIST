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
    w3 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    A = np.exp(x) / sum(np.exp(x))
    return A
    
def forward_propagation(w1, b1, w2, b2, w3, b3, a0):
    z1 = w1.dot(a0) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def ReLU_deriv(x):
    return x > 0

def one_hot(label):
    y = np.zeros((label.size, label.max() + 1))
    y[np.arange(label.size), label] = 1
    y = y.T
    return y

def backward_propagation(z1, a1, z2, a2, a3, w2, w3, a0, y):
    dC_dz3 = a3 - y
    dC_dw3 = 1 / n  * dC_dz3.dot(a2.T)
    dC_db3 = 1 / n  * np.sum(dC_dz3)
    dC_dz2 = w3.T.dot(dC_dz3) * ReLU_deriv(z2)
    dC_dw2 = 1 / n  * dC_dz2.dot(a1.T)
    dC_db2 = 1 / n  * np.sum(dC_dz2)
    dC_dz1 = w2.T.dot(dC_dz2) * ReLU_deriv(z1)
    dC_dw1 = 1 / n  * dC_dz1.dot(a0.T)
    dC_db1 = 1 / n  * np.sum(dC_dz1)
    return dC_dw1, dC_db1, dC_dw2, dC_db2, dC_dw3, dC_db3

def update_parameters(w1, b1, w2, b2, w3, b3, dC_dw1, dC_db1, dC_dw2, dC_db2, dC_dw3, dC_db3, alpha):
    w1 -= alpha * dC_dw1
    b1 -= alpha * dC_db1
    w2 -= alpha * dC_dw2
    b2 -= alpha * dC_db2
    w3 -= alpha * dC_dw3
    b3 -= alpha * dC_db3
    return w1, b1, w2, b2, w3, b3

def predict(a3):
    return np.argmax(a3, 0)

def output(a0, w1, b1, w2, b2, w3, b3):
    _,_,_,_, _, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, a0)
    return a3

def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size
def cross_entropy_loss(p, y):
    loss = -1/n * np.sum(np.sum(y * np.log(p)))
    
    return loss
def gradient_descent(x, y0, alpha, iterations):
    w1, b1, w2, b2, w3, b3 = init_parameters()
    start = time.perf_counter() 
    for i in range(iterations):
        for j in range(n):
            a0 = x[:,j].reshape((784,1))
            y = y0[:,j].reshape((10,1))
            z1, a1, z2, a2, z3, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, a0)
            dw1, db1, dw2, db2, dw3, db3 = backward_propagation(z1, a1, z2, a2, a3, w2, w3, a0, y)
            w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        z1, a1, z2, a2, z3, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, x)
        loss = cross_entropy_loss(a3, y0)
        acc = accuracy(predict(a3), predict(y0))
        loss_history.append(loss)
        accuracy_history.append(acc)
        time_history.append(time.perf_counter()-start)
        if i % 10 == 0:
            print("Epoch: ", i)
            print("Loss:",loss)
            print("Accuracy:", acc)
    return w1, b1, w2, b2, w3, b3

def test(i, w1, b1, w2, b2, w3, b3):
    if i!='r':
        current_image = test_images[:, i,None]
        label = test_labels[i]
        prediction = predict(output(current_image, w1, b1, w2, b2, w3, b3))[0]
    else:
        current_image = np.random.rand(784,1)
        label = 'Null test'
        prediction = np.round(output(current_image, w1, b1, w2, b2, w3, b3),3).flatten()
        prediction = str(max(prediction))
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image)
    plt.title(f"Prediction: {prediction}, Labeled: {label}")
    plt.show()   
w1, b1, w2, b2, w3, b3 = gradient_descent(train_images, one_hot(train_labels), 0.1, int(input("Iterations: ")))    
plt.plot(loss_history)
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.plot(accuracy_history)
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(time_history,loss_history)
plt.title('Loss during training')
plt.xlabel('Seconds')
plt.ylabel('Loss')
plt.show()

plt.plot(time_history,accuracy_history)
plt.title('Accuracy during training')
plt.xlabel('Seconds')
plt.ylabel('Accuracy')
plt.show()  
for i in range(10):
    test(i, w1, b1, w2, b2, w3, b3)
for i in range(5):
    test('r',w1, b1, w2, b2, w3, b3)

unseen = predict(output( test_images,w1, b1, w2, b2, w3, b3))
print("Final Score on unseen images:",accuracy(unseen, test_labels))