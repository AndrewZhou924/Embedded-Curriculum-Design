import argparse
import os
import torch
import json
import PIL
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from numpy import asarray
# model - resnet34
from .Model.nets import resnet34
from .Model.nets import convnet
# dataset
from .DataUtils.load_data import QD_Dataset

def imagePrepareAddCropping(image_path):
    temp_image      = Image.open(image_path).convert('L')
    np_image        = asarray(temp_image)
    (height, width) = np_image.shape # 480, 320
    height_top, height_down = 0, (height-1)
    width_left, width_right = 0, (width-1)
    
    '''
    122400
    81600
    '''
    for h in range(height):
        if sum(np_image[h, :]) != 81600:
            break
        height_top += 1

    for h in reversed(range(height)):
        if sum(np_image[h, :]) != 81600:
            break
        height_down -= 1
        
    for w in range(width):
        if sum(np_image[:, w]) != 122400:
            break
        width_left += 1

    for w in reversed(range(width)):
        if sum(np_image[:, w]) != 122400:
            break
        width_right -= 1

    np_image_crop = np_image[height_top:height_down, width_left:width_right]
    temp_image = Image.fromarray(np.uint8(np_image_crop))
    temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    im = Image.fromarray(np.uint8(temp_image))
    im = PIL.ImageOps.invert(im)
    im = im.resize((28, 28))
    
    return im

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of image classification based on Quick, Draw! data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', '-root', type=str, default='Dataset',
                        help='root for the dataset directory.')
    parser.add_argument('--image_size', '-size', type=int, default=28,
                        help='the size of the input image.')

    # training
    parser.add_argument('--epochs', '-e', type=int, default=1000,
                        help='number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=512, help='batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.01, help='the learningrate.')
    parser.add_argument('--momentum', '-mo', type=float,
                        default=0.9, help='momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float,
                        default=5e-4, help='L2 penalty weight decay.')
    parser.add_argument('--lr_decay_step', '-lrs',
                        type=int, nargs='*', default=[12, 20])
    parser.add_argument('--gamma', '-g', type=float, default=0.1,
                        help='lr is multiplied by gamma on step defined above.')
    parser.add_argument('--ngpu', type=int,
                        default=1, help='0 or less for CPU.')
    parser.add_argument('--model', '-m', type=str,
                        default='resnet34', help='choose the model.')
    parser.add_argument('--Pretrain', type=str,
                        default=None)

    # testing
    parser.add_argument('--test_bs', '-tb', type=int,
                        default=256, help='test batch size.')
    parser.add_argument('--img', type=str, default='./testData/test_0_gt_61.jpg')
    # checkpoint
    parser.add_argument('--save_dir', '-s', type=str,
                        default='./Checkpoints', help='directory for saving checkpoints')
    # for log info
    parser.add_argument('--log', type=str, default='./',
                        help='path of the log info.')

    args = parser.parse_args()
    return args

def getQDmodel():
    args     = parse_args()
    args.Pretrain = './quickDraw/Checkpoints/model.pytorch'
    net = resnet34(100)
    checkpoint = torch.load(args.Pretrain)
    net.load_state_dict(checkpoint) 
    net.cuda()
    net.eval()
    
    return net


def QDinference(imgPath, net=None):
    args     = parse_args()
    args.img = imgPath
    args.Pretrain = './quickDraw/Checkpoints/model.pytorch'
    
    # prepare network
    if net == None:
        net = resnet34(100)
        checkpoint = torch.load(args.Pretrain)
        net.load_state_dict(checkpoint) 
        net.cuda()
        net.eval()

    classes = []
    with open("./quickDraw/DataUtils/class_names.txt", "r") as f:
        for line in f.readlines():
            cls  = line.split('\t')[0].split(' ')[-1]
            classes.append(cls)

    # prepare test data
    data = Image.open(args.img).convert('L')
    data = PIL.ImageOps.invert(data)
    data = data.resize((args.image_size, args.image_size))
    data = transforms.ToTensor()(data)
    data = torch.autograd.Variable(data.cuda())
    data = data.view(-1, 1, args.image_size, args.image_size)
    # data /= 255.0
    
    # inference
    output   = net(data)
    pred     = int(output.data.max(1)[1])
    pred_cls = classes[pred]

    # print("*"*50)
    # print("==> Test result:")
    # print("==> imgPath: {}, pred: {}, cls: {}".format(args.img, pred, pred_cls))
    
    return [pred, pred_cls]
    
if __name__ == '__main__':
    imgPath = './quickDraw/testData/new/cat.jpg'
    [pred, pred_cls] = QDinference(imgPath)