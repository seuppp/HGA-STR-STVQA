"""
该脚本用于抽取ocr的cnn特征（512D），conda环境为server4的sma
1、读取gt注释文件
2、遍历ocr，根据bbox对原图进行裁剪得到新图，将新图传入模型，得到512D的cnn特征
3、将特征存入原注释文件

执行命令
cd /nfs/users/hongbo/HGA-STR
CUDA_VISIBLE_DEVICES=3 python pre_img.py
"""

import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
import numpy as np
import cv2
from models.model import MODEL
from models.transformer.Constants import UNK, PAD, BOS, EOS, PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD
import os, sys, pdb
from tqdm import tqdm

# IMDB_FILE = "/nfs/users/hongbo/SMA/code/data/imdb/imdb_noempty_train.npy"
# IMDB_FILE = "/nfs/users/hongbo/SMA/code/data/imdb/imdb_noempty_val.npy"
# IMDB_FILE = "/nfs/users/hongbo/SMA/code/data/imdb/imdb_noempty_test.npy"

# IMDB_FILE = "/nfs/users/hongbo/CRN_env/data/crn_textvqa/imdb/imdb_train_ppocr.npy"
# IMDB_FILE = "/nfs/users/hongbo/CRN_env/data/crn_textvqa/imdb/imdb_val_ppocr.npy"
IMDB_FILE = "/nfs/users/hongbo/CRN_env/data/crn_textvqa/imdb/imdb_test_ppocr.npy"  # 自身无元数据

IMAGE_DIR = "/nfs/users/hongbo/datasets/textvqa"

model_path = "/nfs/users/hongbo/HGA-STR/trained_model.pth"

alphabet = '0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~'
n_bm = 5
imgW = 160
imgH = 48
nclass = len(alphabet.split(' '))  # 90

# 加载模型
MODEL = MODEL(n_bm, nclass)
MODEL = MODEL.cuda()
# print(MODEL)  # 模型最后一层是个Linear(in_features=1024, out_features=94, bias=False) BOS是90 EOS是91 PAD是92 UNK是93
# 加载模型权重
state_dict = torch.load(model_path)
MODEL_state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # remove `module.`
    MODEL_state_dict_rename[name] = v
MODEL.load_state_dict(MODEL_state_dict_rename)

for p in MODEL.parameters():
    p.requires_grad = False
MODEL.eval()

converter = utils.strLabelConverterForAttention(alphabet, ' ')
transformer = dataset.resizeNormalize((imgW, imgH))

text = torch.LongTensor(1 * 5)
length = torch.IntTensor(1)
text = Variable(text)
length = Variable(length)

max_iter = 35
t, l = converter.encode('0' * max_iter)
utils.loadData(text, t)
utils.loadData(length, l)

# preds = MODEL(image, length, text, text, test=True, cpu_texts='')[0]
# pred = converter.decode(preds.data, length.data + 5)
# pred = pred.split(' ')[0]
# print('################# Answer: '+pred)

imdb = np.load(IMDB_FILE, allow_pickle=True)[1:]
for info in tqdm(imdb):
    image_path = info["image_path"]  # test/6abb8ddf5bd0ac6d.jpg
    path = os.path.join(IMAGE_DIR, image_path)
    image = Image.open(path).convert('RGB')  # 原始图片

    ocr_normalized_boxes = info["ocr_normalized_boxes"]  # n个ocr

    ocr_cnn_feature = []
    print(image_path)
    for bbox in ocr_normalized_boxes:
        lx = bbox[0]
        ly = bbox[1]
        rx = bbox[2]
        ry = bbox[3]
        w = info["image_width"]
        h = info["image_height"]
        ori_bbox = (lx * w, ly * h, rx * w, ry * h)

        crop_image = image.crop(ori_bbox)
        crop_image = transformer(crop_image)  # 3 48 160 固定的
        crop_image = crop_image.cuda()
        crop_image = crop_image.view(1, *crop_image.size())  # 1 3 48 160
        crop_image = Variable(crop_image)

        # cnn特征抽取
        cnn_feature = MODEL(crop_image, length, text, text, test=True, cpu_texts='')
        cnn_feature = cnn_feature.squeeze(dim=0)
        cnn_feature = cnn_feature.to('cpu').numpy()
        ocr_cnn_feature.append(cnn_feature)

    # 已经获取该图片中所有ocr的cnn特征
    info["ocr_cnn_feature"] = ocr_cnn_feature

# 写回原注释文件
np.save(IMDB_FILE, imdb)
