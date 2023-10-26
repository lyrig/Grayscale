import torch
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import SimpleITK as itk
import PIL.Image as Image
import imageio

from tensorboardX import SummaryWriter


def read_img_astensor(file_path, cuda_device=0):
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(file_path).convert('L')
    img = trans(img)
    img = img.cuda(cuda_device)
    return img.squeeze()

def read_img_asnp(file_path):
    img = Image.open(file_path).convert('I')
    img = np.array(img, dtype=np.uint8)
    return img
    
def save_img_tensor(path, img):
    imageio.imwrite(path, img)


def normalize(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))


# =====线性变换======

def indentity(img):
    # 恒等变换
    return img

def picture_inversion(img, L=256):
    # 图像反转
    ret = np.full_like(img, L-1)
    ret = ret - img
    return ret

# 对数变换

def log_transform(img, c=1):
    # 指数变换
    t = c * np.log(1 + img)
    return t

def antilog_transform(img):
    # 反对数变换
    return np.power(10, img)


# 幂律

def gamma_transform(img, c=1, gamma=5):
    return c * np.power(img, gamma)

def nsqrt_transform(img, c=1, gamma=2):
    return c * np.power(img, 1/gamma)


# =============灰度级分层=================
def logarithm_transformation(img, left=50, right=60):
    img_data = np.array(img)
    a = np.shape(img_data)
    new_img = [[] for _ in range(a[0])]
    func = lambda data : power_lawx(data, left, right)
    for i in range(a[0]):
        for j in range(a[1]):
            data = img_data[i][j]
            new_data = func(data)
            new_img[i].append(new_data)
    return new_img

def power_lawx(data, left, right, default_true=255, default_false=0):
    #
    # data 用于变换的数字
    # left, right  提取的区间，区间为闭区间[left, right]
    # default_true  区间内的值采用该值填充
    # default_false  非区间内的值采用该值填充，可填写具体数字或'stable'，后者采取原值
    #
    #
    #
    new_data = 0
    if data >= left and data <= right:
        new_data=default_true
    else:
        if default_false == 'stable':
            new_data=data
        else:
            new_data = default_false
    return new_data

def power_law1(data):
    new_data = 0
    if data >= 150:
        new_data = 255
    else:
        new_data = 0
    return new_data

def power_law2(data):
    new_data = []
    if data >= 150:
        new_data = 255
    else:
        new_data = data
    return new_data


# ===========⽐特平⾯分层=============

def Bitplane_stratification(img, debug=False):
    ret_img = np.full(shape=(8, img.shape[0], img.shape[1]), fill_value=1, dtype=np.uint8)
    for i in range(8):
        if debug:
            print(np.bitwise_and(np.right_shift(img, i), ret_img[i]))
            break
        ret_img[i] = np.bitwise_and(np.right_shift(img, i), ret_img[i])
        print(ret_img[i])
    # print(np.right_shift(img, 0))
    return ret_img

def Bitplane_construct(Bitplane):
    ret_img = np.zeros(shape=(Bitplane.shape[1], Bitplane.shape[2]), dtype=np.uint8)
    for i in range(8):
        value = np.left_shift(Bitplane[i], i)
        ret_img += value
    return ret_img


# ===========直方图处理===============
# 注：目前只集成的函数（直方图规定化）只能规定到某个图像，想要自定义直方图，在compute_histogram一步替代即可。

def compute_histogram(img, normal=True):
    number, counts = np.unique(img, return_counts=True)
    # print(number)
    # print(np.sum(counts))
    if normal:
        counts = counts/np.sum(counts)
    # print()
    plt.figure()
    plt.bar(x=[i for i in range(len(number))], height=counts)
    plt.savefig('./hist.png')
    print()
    np.savez('./hist.npz', number, counts)
    return {'number':number, 'counts':counts}

def compute_histogram_equalization(hist, need_per=False, T=256):
    if need_per:
        hist['counts'] = hist['counts']/np.sum(hist['counts'])
    
    color_list=[]
    raw_color_list = []

    for s in range(T):
        color = 0.0
        for i in range(len(hist['number'])):
            if hist['number'][i] > s:
                break
            color += hist['counts'][i]

        color *= T-1
        raw_color_list.append(color)
        color_list.append(int(color))

    return color_list,raw_color_list


def histogram_equalization(img):
    hist = compute_histogram(img)
    new_color, _ = compute_histogram_equalization(hist)

    func = lambda x : new_color[x]
    new_img = [[] for i in range(img.shape[0])]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            nc = func(img[x][y])
            new_img[x].append(nc)
        
    return new_img

def compute_histogram_regularization(img, target, T=256, method='single'):
    input_hist = compute_histogram(img)
    input_std, input_raw = compute_histogram_equalization(input_hist)
    # print(input_raw)

    target_hist = compute_histogram(target)
    target_std, target_raw = compute_histogram_equalization(target_hist)
    print(target_raw)

    reflect = []

    if method=='single':
        for i_ids in range(T):
            min = 999999
            for t_ids in range(T):
                # print(abs(input_raw[i_ids] - target_raw[t_ids]))
                if abs(input_raw[i_ids] - target_raw[t_ids]) > min:
                    reflect.append(t_ids-1)
                    # print(f'{abs(input_raw[i_ids] - target_raw[t_ids])}, {min}, {target_raw[t_ids]}')
                    break
                else:
                    min = abs(input_raw[i_ids] - target_raw[t_ids])
                
                if t_ids == T-1:
                    reflect.append(t_ids-1)
    elif method=='group':
        for i_ids in range(T):
            min = 999999
            for t_ids in range(0, i_ids+1):
                if abs(input_raw[i_ids] - target_raw[t_ids]) >= min:
                    reflect.append(t_ids)
                    break
                else:
                    min = abs(input_raw[i_ids] - target_raw[t_ids])
        
    return reflect

def histogram_regularization(img, target, T=256, method='single'):
    reflect = compute_histogram_regularization(img, target, method=method)
    print(reflect)
    func = lambda x : reflect[x]
    new_img = [[] for i in range(img.shape[0])]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            nc = func(img[x][y])
            new_img[x].append(nc)
        
    return new_img


if __name__ == '__main__':
    dir = '/home/vision/diska4/shy/NerfDiff/data/LIDC/XRay-Coronal-X2CT'
    path = os.path.join(dir, '0001.png')
    img = read_img_asnp('/home/vision/diska4/shy/Grayscale/equ.png')

    # ret = picture_inversion(img)

    compute_histogram(img)
    ret = logarithm_transformation(img, left=130, right=255)
    # t = np.load('./hist.npz').files

    # ret = Bitplane_stratification(img)

    # save_img_tensor('./tg.png', img)
    # for i in range(8):
    #     save_img_tensor(f'./test{i}.png', ret[i] * 255)


    # ret = Bitplane_construct(ret)
    # print(ret)
    # print(ret.shape)
    save_img_tensor(f'./log.png', ret)

