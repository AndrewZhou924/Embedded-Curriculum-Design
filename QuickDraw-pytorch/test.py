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

# model - resnet34
from Model.nets import resnet34
from Model.nets import convnet

# dataset
from DataUtils.load_data import QD_Dataset

if __name__ == '__main__':
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

    if not os.path.isdir(args.log):
        os.makedirs(args.log)

    # log = open(os.path.join(args.log, 'log.txt'), 'w')
    # state = {k: v for k, v in args._get_kwargs()}
    # log.write(json.dumps(state)+'\n')

    # Init save directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    net = None
    num_classes = 100
    if args.model == 'resnet34':
        net = resnet34(num_classes)
    elif args.model == 'convnet':
        net = convnet(num_classes)

    if args.Pretrain != None:
        checkpoint = torch.load(args.Pretrain)
        net.load_state_dict(checkpoint)
        print("==> load pretrained model from: ", args.Pretrain)
        
    if args.ngpu > 1:
        net = nn.DataParallel(net)

    if args.ngpu > 0:
        net.cuda()

    net.eval()

    classes = []
    with open("./DataUtils/class_names.txt", "r") as f:
        for line in f.readlines():
            cls  = line.split('\t')[0].split(' ')[-1]
            classes.append(cls)

    # prepare test data
    data = Image.open(args.img).convert('L')
    data = PIL.ImageOps.invert(data)
    
    # print(np.array(data)[-1])
    data = data.resize((args.image_size, args.image_size))

    # data.save('./resize_cat.jpg')

    data = transforms.ToTensor()(data)
    data = torch.autograd.Variable(data.cuda())
    data = data.view(-1, 1, args.image_size, args.image_size)
    # data /= 255.0
    
    # inference
    output   = net(data)
    pred     = int(output.data.max(1)[1])
    pred_cls = classes[pred]

    print("*"*50)
    print("==> Test result:")
    print("==> imgPath: {}, pred: {}, cls: {}".format(args.img, pred, pred_cls))

    # batch
    