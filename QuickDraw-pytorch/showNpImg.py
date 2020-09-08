import numpy as np
from   PIL import Image

data = np.load('./Data/apple.npy')
for i in range(10):
    img = data[i].reshape((28,28))
    img = Image.fromarray(np.uint8(img))
    img.show()
    
    a = input()