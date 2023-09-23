
import cv2
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from nn import output

class Recogniser:                 
    def __init__(self, resolution, cap):
        self.width, self.height = resolution
        self.waiting = False
        self.final =[]
        self.start = []
        self.end = []
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.start_end)
        print("Click the spacebar to freeze.")
        while(True):
            ret, self.frame = cap.read()
            self.frame = cv2.resize(self.frame, (self.height, self.width))
            self.colour_frame = self.frame 
            white = (255, 255, 255)
            dark_white = (50, 50, 50)
            mask = cv2.inRange(self.frame, dark_white, white)
            
            self.frame = cv2.bitwise_and(self.frame, self.frame, mask=mask)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            lower_thres = 80
            thresh, self.frame = cv2.threshold(self.frame, lower_thres, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow("frame", self.frame)
            if cv2.waitKey(1) & 0xFF == 32:
                print("Frozen!")
                self.f0 = self.colour_frame
                self.waiting = True
                print("Put the image in the frame")
                cv2.waitKey(0)
                break
        
        return
    def start_end(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN and self.waiting:
            self.colour_frame = deepcopy(self.f0)          
            cv2.line(self.colour_frame, (x, y), (x+self.width//2, y), (0, 255, 0), thickness=2)
            cv2.line(self.colour_frame, (x, y), (x, y+self.width//2), (0, 255, 0), thickness=2)
            cv2.line(self.colour_frame, (x+self.width//2, y), (x+self.width//2, y+self.width//2), (0, 255, 0), thickness=2)
            cv2.line(self.colour_frame, (x, y+self.width//2), (x+self.width//2, y+self.width//2), (0, 255, 0), thickness=2)
            cv2.imshow("frame", self.colour_frame)
            self.final = deepcopy(self.frame[y:y+self.width//2, x:x+self.width//2])
CAMERA = 0
resolution = (200, 200)

w1,w2,w3,b1,b2,b3 = np.load('saved_network/w1.npy'),np.load('saved_network/w2.npy'),np.load('saved_network/w3.npy'),np.load('saved_network/b1.npy'),np.load('saved_network/b2.npy'),np.load('saved_network/b3.npy')

cam = cv2.VideoCapture(CAMERA)
while True:
    recogniser = Recogniser(resolution, cam)
    frame = recogniser.final
    while frame.any() == None:
        frame = recogniser.final
    frame = frame /  255
    height, width = frame.shape
    data = np.average(np.split(np.average(np.split(frame, width // (height//20), axis=1), axis=-1), height//(height//20), axis=1), axis=-1)
    data = np.ceil(data)
    drawn_image = np.pad(data,[(4,4),(4,4)],mode='constant')
    image = drawn_image.reshape((784, 1))
    prob = output(image,w1, b1, w2, b2, w3, b3)
    guess = np.argmax(prob,0)
    fig = plt.figure()
    plt.gray()
    plt.subplot(1,2,1)
    plt.title(f"Guessed: {guess[0]}")
    
    plt.imshow(drawn_image)
    
    plt.subplot(1,2,2)
    prob = prob.reshape(10,)
    plt.bar(list(range(len(prob))), prob.tolist(), color='lightblue')
    plt.xticks(np.arange(0, 10))
    plt.ylim([0, 1])
    plt.title('Prediction')
    
    fig.tight_layout()
    plt.show()



    

