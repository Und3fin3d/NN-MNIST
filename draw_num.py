import numpy as np
from matplotlib import pyplot as plt
from nn import output
def draw_digit():
    fig, ax = plt.subplots(figsize = (4,4))
    pixels = np.zeros((20, 20))
    def draw(event):
        if event.button == 1 and (event.xdata!=None) and (event.ydata!=None):
            x0 = int(event.xdata)
            y0 = int(event.ydata)
            for y in range(y0-1,y0+1):
                for x in range(x0-1,x0+1):
                    pixels[y, x] = 1
            """  
            for x in range(x0-1,x0+1):
                if pixels[y0+1, x] !=1:
                    pixels[y0+1, x] = 1
            for x in range(x0-1,x0+1):
                if pixels[y0-1, x] !=1:
                    pixels[y0-1, x] = 1
            for y in range(y0-1,y0+1):
                if pixels[y, x0+1] != 1:
                    pixels[y, x0+1]  = 1
            for y in range(y0-1,y0+1):
                if pixels[y, x0-1]!= 1:
                    pixels[y, x0-1] = 1
            
            """

            
            
            ax.imshow(pixels, cmap='gray')
            fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', draw)
    plt.title("Draw a digit, then close the window")
    plt.gray()
    plt.imshow(pixels)
    plt.show()

    return pixels



w1,w2,w3,b1,b2,b3 = np.load('saved_network/w1.npy'),np.load('saved_network/w2.npy'),np.load('saved_network/w3.npy'),np.load('saved_network/b1.npy'),np.load('saved_network/b2.npy'),np.load('saved_network/b3.npy')
while True:
    drawn_image = np.pad(draw_digit(),[(4,4),(4,4)],mode='constant')
    image = drawn_image.reshape((784, 1))
    prob = output(image,w1, b1, w2, b2, w3, b3)
    guess = np.argmax(prob,0)
    fig = plt.figure()
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