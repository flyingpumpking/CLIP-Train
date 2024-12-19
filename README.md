# README

+ 存放CLIP模型训练所用的图片和对应的标签

  + 路径：`CLIP-train/datasets/part-images`

+ 训练函数

  + `train.py`

    + 参数：

      | 参数名        | 位置（行） | 功能                                             |
      | ------------- | ---------- | ------------------------------------------------ |
      | `EPOCH`       | 98         | 训练轮数                                         |
      | `BATCH_SIZE`  | 99         | 每轮训练样本大小                                 |
      | `img_folder`  | 100        | 训练用的图片文件夹路径                           |
      | `txt_folder`  | 101        | 训练用的图片对应标签文件路径                     |
      | `img_dic`     | 102        | 标签对应文字字典                                 |
      | `weight_img`  | 122        | 损失函数图片权重                                 |
      | `weight_text` | 123        | 损失函数文字权重（weight_text = 1 - weight_img） |
      | `save_folder` | 164        | 训练模型存储文件夹路径                           |