# Embedded-Curriculum-Design

Members: [Ziqing Pan](<https://github.com/OzwardPenrose>), [Yongwei Wang](<https://github.com/canian1999>), [Zhanke Zhou](<https://github.com/AndrewZhou924>)



## Usage

Visualization

```
python3 visualization.py --file ./data/output.txt --show 1 --save 1
```



## Issue

CUDA related

- can't find cuda lib

```
sudo ldconfig /usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
```

