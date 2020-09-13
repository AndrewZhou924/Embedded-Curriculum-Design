# Embedded-Curriculum-Design

*Advisor: Guohui Zhong*

*Members: [Ziqing Pan](<https://github.com/OzwardPenrose>), [Yongwei Wang](<https://github.com/canian1999>), [Zhanke Zhou](<https://github.com/AndrewZhou924>)*



## Usage

Quick start

```
# character recognition
./tools/run_cnn_en.sh

# draw and guess 
./tools/run_draw.sh

# image generation via GAN
./tools/run_gan.sh
```

Visualization

```
python3 visualization.py --file ./data/output.txt --show 1 --save 1
```

English character & number recognition

```
python3 ./Embedded-Curriculum-Design/cnnRecognition/testEasyOcr.py --img ./Embedded-Curriculum-Design/cnnRecognition/app/image/en_words/interesting_clean2.jpg
```

Chinese character recognition

````
# single img
python3 ./Embedded-Curriculum-Design/cnnRecognition/app/CNNinference.py --img ./Embedded-Curriculum-Design/cnnRecognition/app/image/pred1.png

# a folder of imgs
python3 ./Embedded-Curriculum-Design/cnnRecognition/app/CNNinference.py --folder ./Embedded-Curriculum-Design/cnnRecognition/app/image/all_single_cnn/
````

Your draw and I guess

```
python3 test_ori.py --Pretrain ./Checkpoints/model.pytorch --img ./testData/test_1_gt_36.jpg
```



## Issue

CUDA related

- can't find cuda lib

```
sudo ldconfig /usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
```

