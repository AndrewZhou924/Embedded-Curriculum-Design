import os
import shutil
import sys
import inspect

currentdir  = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataroot    = currentdir + '/GAN_tmp_img/'
results_dir = currentdir + '/GAN_tmp_results/'  
  

# childrendir  = currentdir + "/pix2pix"
# print(childrendir)
# sys.path.insert(0, childrendir)

def GAN_generate(img_path):
    if os.path.exists(dataroot):
        shutil.rmtree(dataroot)
    os.mkdir(dataroot)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    cp_to_path = dataroot + img_path.split('/')[-1]
    shutil.copy(img_path, cp_to_path)
    os.system('python3 ./pix2pix/GAN_inference.py')
    # os.system('sh ./pix2pix/run.sh')
    
    paths = img_path.split('/')[-1].split('.')
    resultPath = results_dir + 'edges2shoes_pretrained/test_latest/images/' + paths[0] + "_fake." + paths[-1]
    
    return resultPath
    
if __name__ == "__main__":
    # img_path = './images/test.png'
    img_path   = currentdir + '/images/test.png'
    resultPath = GAN_generate(img_path)
    print("resultPath:", resultPath)