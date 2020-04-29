# 介绍
本文训练相关代码基于[yolact](https://github.com/dbolya/yolact)
# 安装
pip install pytorch torchvision
pip install cython
pip install opencv-python pillow pycocotools matplotlib 

# 训练
将COCO stuff的数据集放入data/coco目录，之后data/coco下目录结构应该如下：
├── annotations
│   ├── deprecated-challenge2017
│   ├── stuff_train2017.json
│   ├── stuff_train2017_pixelmaps.zip
│   ├── stuff_val2017.json
│   └── stuff_val2017_pixelmaps.zip
├── train2017
└── val2017
然后执行：
  sh train.sh yolact_stuffwallonly_res50_config 32
其中32为4GPU * 8batchsize，根据实际情况调整

# 测试
1. 从待测视频(如abc.mp4)第一帧截取marker，存在anchor/abc下
2. 将待测视频放入video_input目录下
3. 执行
  CUDA_VISIBLE_DEVICES=1 python eval.py --trained_model=weights/yolact_stuff_res50_88_200000.pth --score_threshold=0.3 --top_k=5 --video_multiframe=1 --video=video_input::video_output
该命令会将依次对video_input目录下的视频进行处理。
训练好的模型参数可以从这里下载：[百度网盘](https://pan.baidu.com/s/1bKgR6skyzrVqn9nwccIrIQ)  提取码：hryc 