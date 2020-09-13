#!/usr/bin/python3
# coding:utf-8
from   flask import render_template,json,jsonify,request
import base64
import tensorflow as tf
import numpy as np
import datetime
import tensorflow.contrib.slim as slim
import pickle
import argparse
import glob
import easyocr
from   PIL import Image,ImageFont, ImageDraw
from   numpy import asarray

__global_times         = 0 
__chinese_word_count   = 3755     # 常见汉字个数
# __projectRoot          = './Embedded-Curriculum-Design/cnnRecognition/'
__projectRoot          = './cnnRecognition/'
__checkpoint_dir       = __projectRoot + 'app/train_model/checkpoint/'     # 模型文件路径
__code_to_chinese_file = __projectRoot + 'app/train_model/code_word.pkl'   # 文字和对应的编码
__test_image_file      = __projectRoot + 'app/image/test.png'              # 测试图片
__pred1_image_file     = __projectRoot + 'app/image/pred1.png'             # 预测结果1图片
__pred2_image_file     = __projectRoot + 'app/image/pred2.png'             # 预测结果2图片
__pred3_image_file     = __projectRoot + 'app/image/pred3.png'             # 预测结果3图片

'''
    Usage:
    python3 ./Embedded-Curriculum-Design/cnnRecognition/app/CNNinference.py --img ./Embedded-Curriculum-Design/cnnRecognition/app/image/pred1.png
    python3 ./Embedded-Curriculum-Design/cnnRecognition/app/CNNinference.py --folder ./Embedded-Curriculum-Design/cnnRecognition/app/image/all_single_cnn/
'''

def parse_args():
    parser = argparse.ArgumentParser(description='visualization script for embedding course design')
    parser.add_argument('--img', dest='img',
            default='./data/output.txt', type=str)
    parser.add_argument('--folder', dest='folder',
            default=None, type=str)
    args = parser.parse_args()
    return args

# 构建一个三个卷积(3x3) + 三个最大池化层(2x2) +  两个FC层
def buildCnn(top_k):
    # with tf.device('/cpu:0'):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch') # image_size 64x64
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')  # image_size 62x62
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')      # image_size 31x31
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')   # image_size 29x29
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')      # image_size 15x15
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')      # image_size 13x13
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')      # image_size 7x7

    flatten = slim.flatten(max_pool_3)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')  # 激活函数tanh
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob),__chinese_word_count, activation_fn=None,scope='fc2') # 无激活函数
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) # softmax
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32)) # 计算准确率

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True) #
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step) # 自动调节学习率的随机梯度下降算法训练模型
    probabilities = tf.nn.softmax(logits) #

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {
        'images': images,
        'labels': labels,
        'keep_prob': keep_prob,
        'top_k': top_k,
        'global_step': global_step,
        'train_op': train_op,
        'loss': loss,
        'accuracy': accuracy,
        'accuracy_top_k': accuracy_in_top_k,
        'merged_summary_op': merged_summary_op,
        'predicted_distribution': probabilities,
        'predicted_index_top_k': predicted_index_top_k,
        'predicted_val_top_k': predicted_val_top_k
    }

def predictPrepare():
    sess  = tf.Session()
    graph = buildCnn(top_k=3)
    saver = tf.train.Saver()
    ckpt  = tf.train.latest_checkpoint(__checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    return graph, sess

def imagePrepare(image_path):
    temp_image = Image.open(image_path).convert('L')
    temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
    
    # temp_image.show()
    # a = input()
    
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    return temp_image

def createImage(predword,imagepath):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fonts = ImageFont.truetype(__projectRoot + "/app/static/fonts/msyh.ttc",36,encoding='utf-8')
    dr.text((15, 10), predword,font=fonts,fill="#000000")
    im.save(imagepath)

'''
    API for single image process
'''
def chineseRecognizeSingleImage(imgPath=__test_image_file):
    global __global_times
    
    oldtime = datetime.datetime.now()
    
    # prepare model
    __graph1, __sess1 = predictPrepare() 
    
    # prepare data
    temp_image = imagePrepareAddCropping(imgPath)
    print("==> [chineseRecognizeSingleImage] image.shape: ", temp_image.shape)
    
    # inference
    predict_val, predict_index = __sess1.run([__graph1['predicted_val_top_k'], __graph1['predicted_index_top_k']],
                                              feed_dict={__graph1['images']: temp_image, __graph1['keep_prob']: 1.0}) # 预测top3的汉字编码以及相应的准确率
    
    # process result
    with open(__code_to_chinese_file, 'rb') as f2:
        word_dict = pickle.load(f2) # 汉字和编码对照字典
    createImage(word_dict[predict_index[0][0]], __pred1_image_file) # 生成准确率top1的汉字图片
    createImage(word_dict[predict_index[0][1]], __pred2_image_file)
    createImage(word_dict[predict_index[0][2]], __pred3_image_file)
    
    # 将识别图片转码传给前端，并带上对应的准确率
    with open(__pred1_image_file, 'rb') as fin:
        image1_data = fin.read()
        pred1_image = base64.b64encode(image1_data)
    with open(__pred2_image_file, 'rb') as fin:
        image2_data = fin.read()
        pred2_image = base64.b64encode(image2_data)
    with open(__pred3_image_file, 'rb') as fin:
        image3_data = fin.read()
        pred3_image = base64.b64encode(image3_data)
        
    info = dict()
    info['pred1_image']    = "data:image/jpg;base64," + pred1_image.decode()
    info['pred1_accuracy'] = str('{:.2%}'.format(predict_val[0][0]))
    info['pred1_acc_float']= float(predict_val[0][0])
    info['pred1_cnn']      = word_dict[predict_index[0][0]]
    info['pred1_image']    = __pred1_image_file
    
    info['pred2_image']    = "data:image/jpg;base64," + pred2_image.decode()
    info['pred2_accuracy'] = str('{:.2%}'.format(predict_val[0][1]))
    info['pred2_cnn']      = word_dict[predict_index[0][1]]
    info['pred2_image']    = __pred2_image_file
                                       
    info['pred3_image']    = "data:image/jpg;base64," + pred3_image.decode()
    info['pred3_accuracy'] = str('{:.2%}'.format(predict_val[0][2]))
    info['pred3_cnn']      = word_dict[predict_index[0][2]]
    info['pred3_image']    = __pred3_image_file

    newtime = datetime.datetime.now()
    print ('相差：%s'%(newtime-oldtime))
    print ('相差：%s微秒'%(newtime-oldtime).microseconds)
    print ('相差：%s秒'%(newtime-oldtime).seconds)
    
    return info

def chineseRecognizeSingleImageWithSess(__graph1, __sess1, imgPath=__test_image_file):
    global __global_times
    oldtime = datetime.datetime.now()
    
    # prepare data
    temp_image = imagePrepareAddCropping(imgPath)
    
    # inference
    predict_val, predict_index = __sess1.run([__graph1['predicted_val_top_k'], __graph1['predicted_index_top_k']],
                                              feed_dict={__graph1['images']: temp_image, __graph1['keep_prob']: 1.0}) # 预测top3的汉字编码以及相应的准确率
    
    # process result
    with open(__code_to_chinese_file, 'rb') as f2:
        word_dict = pickle.load(f2) # 汉字和编码对照字典
    createImage(word_dict[predict_index[0][0]], __pred1_image_file) # 生成准确率top1的汉字图片
    createImage(word_dict[predict_index[0][1]], __pred2_image_file)
    createImage(word_dict[predict_index[0][2]], __pred3_image_file)
    
    # 将识别图片转码传给前端，并带上对应的准确率
    with open(__pred1_image_file, 'rb') as fin:
        image1_data = fin.read()
        pred1_image = base64.b64encode(image1_data)
    with open(__pred2_image_file, 'rb') as fin:
        image2_data = fin.read()
        pred2_image = base64.b64encode(image2_data)
    with open(__pred3_image_file, 'rb') as fin:
        image3_data = fin.read()
        pred3_image = base64.b64encode(image3_data)
        
    info = dict()
    info['pred1_image']    = "data:image/jpg;base64," + pred1_image.decode()
    info['pred1_accuracy'] = str('{:.2%}'.format(predict_val[0][0]))
    info['pred1_acc_float']= float(predict_val[0][0])
    info['pred1_cnn']      = word_dict[predict_index[0][0]]
    info['pred1_image']    = __pred1_image_file
    
    info['pred2_image']    = "data:image/jpg;base64," + pred2_image.decode()
    info['pred2_accuracy'] = str('{:.2%}'.format(predict_val[0][1]))
    info['pred2_cnn']      = word_dict[predict_index[0][1]]
                                       
    info['pred3_image']    = "data:image/jpg;base64," + pred3_image.decode()
    info['pred3_accuracy'] = str('{:.2%}'.format(predict_val[0][2]))
    info['pred3_cnn']      = word_dict[predict_index[0][2]]

    newtime = datetime.datetime.now()
    '''
    print ('相差：%s'%(newtime-oldtime))
    print ('相差：%s微秒'%(newtime-oldtime).microseconds)
    print ('相差：%s秒'%(newtime-oldtime).seconds)
    '''
    
    return info

'''
    crop blank parts in single image
'''
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
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    
    return temp_image

def recognizeEngAndNumber(imgPath):
    # need to run only once to load model into memory
    reader = easyocr.Reader(['ch_sim','en']) 
    result = reader.readtext(imgPath)
    return result
     

if __name__ == "__main__":
    args    = parse_args()
    imgPath = args.img
    folder  = args.folder
    
    # inference for images in folder
    if folder != None:
        print(glob.glob(folder + "/*"))
        # prepare model
        __graph1, __sess1 = predictPrepare() 
        for img in glob.glob(folder + "/*"):
            result  = chineseRecognizeSingleImageWithSess(__graph1, __sess1, img)
            print(img, result['pred1_cnn'], result['pred1_accuracy'])
        
        exit()
    
    # inference for single image
    result  = chineseRecognizeSingleImage(imgPath)
    for key,value in result.items():
        if "_image" in key:
            continue    
        print(key,value)
        
    