import easyocr
import argparse

'''
    Usage:
    python3 app/inference.py --file ./app/image/pred1.png
'''

def parse_args():
    parser = argparse.ArgumentParser(description='visualization script for embedding course design')
    parser.add_argument('--img', dest='img',
            default='./app/image/zoomin_1.png', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args   = parse_args()
    reader = easyocr.Reader(['ch_sim','en']) # need to run only once to load model into memory
    # result = reader.readtext(args.img, detail=0)
    result = reader.readtext(args.img)
    print(result)