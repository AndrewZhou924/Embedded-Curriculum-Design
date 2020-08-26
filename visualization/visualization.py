import numpy as np
import argparse
from PIL import Image

'''
    Usage:
    python3 visualization.py --file ./data/output.txt --show 1 --save 1
'''

def parse_args():
    parser = argparse.ArgumentParser(description='visualization script for embedding course design')
    parser.add_argument('--file', dest='file',
            default='./data/output.txt', type=str)
    
    parser.add_argument('--show', dest='show',
            default=1, type=int)
    
    parser.add_argument('--save', dest='save',
            default=1, type=int)

    args = parser.parse_args()
    return args
    
def showImgViaTxtFile(fileName, show=True, save=False):
    f = open(fileName, 'r')
    all_string = ""
    for line in f.readlines(): 
        if '2020' in line:
            continue
        all_string += line.strip('\n')
        
    each_lines = all_string.split('AE 00')[:-1]
    data_np = np.zeros((480, 320, 3))
    
    cnt1 = 0
    cnt2 = 0
    for i,line in enumerate(each_lines):
        split_line = line.split(' ')[:-1]
        for j in range(640)[::2]:
            pixel = split_line[j]+split_line[j+1]

            if pixel=="FFFF":
                data_np[i, int(j/2), :] = np.array([255, 255, 255])
                cnt1 += 1
            else:
                data_np[i, int(j/2), :] = np.array([0,   0,   0])
                cnt2 += 1
    
    print("cnt for write pixel:{}".format(cnt1))
    print("cnt for black pixel:{}".format(cnt2))
    
    im = Image.fromarray(np.uint8(data_np))
    
    if show:
        im.show()
    if save:
        savePath = fileName.replace('.txt', '.jpg')
        im.save(savePath)
        print("==> save to: ", savePath)
    
if __name__=='__main__':
    args = parse_args()
    show = args.show
    save = args.save
    print(args.file, show, save)
    showImgViaTxtFile(args.file, show, save)