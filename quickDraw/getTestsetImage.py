import numpy as np
from PIL import Image

npz_data = np.load('./Dataset/test.npz')
data     = npz_data['data']
target   = npz_data['target']

for i in range(100):
    img = data[i].reshape((28,28))
    img = Image.fromarray(np.uint8(img))
    # img.show()
    # a = input()
    savaPath = "./testData/test_{}_gt_{}.jpg".format(i, int(target[i]))
    img.save(savaPath)
