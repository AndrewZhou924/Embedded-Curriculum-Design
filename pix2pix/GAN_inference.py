import os
import shutil
import sys
import inspect
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

'''
python3 GAN_inference.py --dataroot images/test/ --name edges2shoes_pretrained 
    --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch 
'''
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    # default settings
    currentdir         = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    opt.dataroot       = currentdir + '/GAN_tmp_img/'
    opt.results_dir    = currentdir + '/GAN_tmp_results/'
    opt.name           = 'edges2shoes_pretrained'
    opt.model          = 'test'
    opt.netG           = 'unet_256'
    opt.direction      = 'BtoA'
    opt.dataset_mode   = 'single'
    opt.norm           = 'batch'
    
    # hard-code some parameters for test
    opt.num_threads    = 0       # test code only supports num_threads = 1
    opt.batch_size     = 1       # test code only supports batch_size = 1
    opt.serial_batches = True    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip        = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id     = -1      # no visdom display; the test code saves the results to a HTML file.
    dataset            = create_dataset(opt)    # create a dataset given opt.dataset_mode and other options
    model              = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)                            # regular setup: load and print networks; create schedulers
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        
    # print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    if opt.eval:
        model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        # print(data['A'].shape) # [1, 3, 256, 256]
        # print(data['B'].shape)
        # a = input()
        
        # print(data)
        # for k,v in data.items():
            # print(k, v.shape)
            
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals  = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()      # get image paths
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    
    webpage.save()  # save the HTML
