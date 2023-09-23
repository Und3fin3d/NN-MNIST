import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

#data = pd.read_csv('train.csv')
data = np.zeros((784,4100))
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
    z = np.exp(x) / sum(np.exp(x))
    return z
    
def forward_propagation(w1, b1, w2, b2, w3, b3, a0):
    z1 = np.matmul(w1,a0) + b1
    a1 = ReLU(z1)
    z2 = np.matmul(w2,a1) + b2
    a2 = ReLU(z2)
    z3 = np.matmul(w3,a2) + b3
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
    #note we skip dC/da3 and da3/dz3 
    # as the product of the derivatives of cross entropy loss and softmax functions, dC_dz3, is far more simplified 
    dC_dz3 = a3 - y
    #as we have forward propagated the whole dataset we normalize it by 1/n to make sure the weights are not too big
    #note dz3/dw3 = a2, which is why it is multiplied with dC/dz3 to get dC/dw3
    dC_dw3 = 1 / n  * np.matmul(dC_dz3,a2.T)
    dC_db3 = 1 / n  * np.sum(dC_dz3)#note for biases we take sum as they are not matrices
    #dC/dz2 = dC/dz3 x dz3/da2 x da2/dz2
    #dC/dz2 = dC/dz3 x w3 x reLu'(z2)
    dC_dz2 = np.matmul(w3.T,dC_dz3) * ReLU_deriv(z2)
    #dC/dw2 = dC/dz2 x dz2/dw2
    #dC/dw2 = dC/dz2 x a1 
    dC_dw2 = 1 / n  * np.matmul(dC_dz2, a1.T)
    dC_db2 = 1 / n  * np.sum(dC_dz2)
    #dC/dz1 = dC/dz2 x dz2/da1 x da1/dz1
    #dC/dz2 = dC/dz2 x w2 x reLu'(z1)
    
    dC_dz1 = np.matmul(w2.T, dC_dz2) * ReLU_deriv(z1)
    dC_dw1 = 1 / n  * np.matmul(dC_dz1,a0.T)
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
    return -1/n * np.sum(np.sum(y * np.log(p)))

def gradient_descent(a0, y, alpha, iterations):
    w1, b1, w2, b2, w3, b3 = init_parameters()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_propagation(w1, b1, w2, b2, w3, b3, a0)
        dw1, db1, dw2, db2, dw3, db3 = backward_propagation(z1, a1, z2, a2, a3, w2, w3, a0, y)
        w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Loss:",cross_entropy_loss(a3,y))
            print("Accuracy:", accuracy(predict(a3), predict(y)))
    return w1, b1, w2, b2, w3, b3



def test(i, w1, b1, w2, b2, w3, b3):
    if i!='r':
        current_image = test_images[:, i,None]#change to test
        label = test_labels[i]
        prediction = predict(output(current_image, w1, b1, w2, b2, w3, b3))[0]
    else:
        current_image = np.random.rand(784,1)
        label = 'Null test'
        out = output(current_image, w1, b1, w2, b2, w3, b3)
        prediction = np.round(out,3).flatten()
        prediction = f"Predicted a {predict(out)[0]} with certainty {max(prediction)*100}%"
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image)
    plt.title(f"Prediction: {prediction}, Labeled: {label}")
    plt.show()

"""
start = time.perf_counter()    
w1, b1, w2, b2, w3, b3 = gradient_descent(train_images, one_hot(train_labels), 0.1, int(input("Iterations: ")))
np.save('w1.npy', w1)
np.save('b1.npy', b1)
np.save('w2.npy', w2)
np.save('b2.npy', b2)
np.save('w3.npy', w3)
np.save('b3.npy', b3)        
end = time.perf_counter()    
print(f"Training took {end - start} seconds")    
for i in range(10):
    test(i, w1, b1, w2, b2, w3, b3)
for i in range(5):
    test('r',w1, b1, w2, b2, w3, b3)

unseen = predict(output( test_images,w1, b1, w2, b2, w3, b3))
print("Final Score on unseen images:",accuracy(unseen, test_labels))

"""