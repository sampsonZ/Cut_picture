import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os, time
from scipy.signal import argrelmin, argrelmax

def r_argrelmax(array, folds):
    return argrelmax(array + np.random.random(len(array)) * 1e-5, order = int(len(array) / folds))
def r_argrelmin(array, folds):
    return argrelmin(array + np.random.random(len(array)) * 1e-5, order = int(len(array) / folds))

# cut shadow by diff
def cut_shadow(img, return_bias = False, verbose = False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_fil = cv2.pyrDown(cv2.pyrDown(img))
    h, w = img.shape[:2]
    diff0 = np.diff(img_fil.mean(0))
    stride = int(len(diff0) / 3)
    left = r_argrelmax(diff0[:stride], 2)[0][0]
    right = r_argrelmin(diff0[-stride:], 2)[0][-1] + (2 * stride)
    
    diff1 = np.diff(img_fil.mean(1))
    stride = int(len(diff1) / 3)
    top = r_argrelmax(diff1[:stride], 2)[0][0]
    down = r_argrelmin(diff1[-stride:], 2)[0][-1] + (2 * stride)
    top, down, left, right = 4 * np.array([top+1, down+1, left+1, right+1])

    if not return_bias:
        if (top > (h * .1)) or (down < (h * .9)) or (left > (w * .05) or right < (w * .95)):
            print('it seems that no cut shadow are needed~')
            if verbose:
                plt.imshow(img)
                for i in [left, right]:
                    plt.axvline(i, c = 'r')
                for i in [top, down]:
                    plt.axhline(i, c = 'r')
                plt.show()
            return img, None
        
        return img[top:down, left:right]
    
    return img[top:down, left:right], [top, left]
        

def get_cut_line(img, row, col, verbose = False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape[:2]
    v_mean = img.mean(0) + 1e-5 * np.random.random(w)
    vlines = argrelmin(v_mean, order = int(w / (col+2)))[0]
    vlines = np.hstack(([0], vlines, [w]))

    h_mean = img.mean(1) + 1e-5 * np.random.random(h)
    hlines = argrelmin(h_mean, order = int(h / (row+2)))[0]
    hlines = np.hstack(([0], hlines, [h]))

    assert len(vlines) == col + 1
    assert len(hlines) == row + 1

    return vlines, hlines

def grid_cut(img, rows, cols, edge_removed = False):
    '''
    按照电池片分界，切割组件图，返回切割线坐标
    :param img_data: 图像数据
    :param rows: 电池片的行数
    :param cols: 电池片列数
    :edge_removed: 默认False，是否无黑边
    :returns: 切割后的图像及其位置，注意返回的位置为电池片图像在大图片中的坐标
    '''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    tic = time.time()
    # 切黑边
    if not edge_removed:
        img_cutshadow, bias = cut_shadow(img, return_bias=True)
    else: 
        img_cutshadow = img.copy()
        bias = [0, 0]
    toc1 = time.time()

    # 获取切割线
    try:
        vlines, hlines = get_cut_line(img_cutshadow, rows, cols)
    except Exception as e:
        print('error in get cut line, use average cut as default.', Exception, e)
        h, w = img_cutshadow.shape[:2]
        vlines = int(w / cols) * np.arange(cols + 1)
        hlines = int(h / rows) * np.arange(rows + 1)
    hlines = hlines + bias[0]
    vlines = vlines + bias[1]
    toc2 = time.time()

    # 切图
    cut_images = {}
    for h_cut in zip(hlines[:-1], hlines[1:]):
        for v_cut in zip(vlines[:-1], vlines[1:]):
            key = (h_cut[0], v_cut[0])
            img_cut = img[h_cut[0]:h_cut[1], v_cut[0]:v_cut[1]]
            cut_images[key] = img_cut
    toc3 = time.time()

    print('cut shadow:%.4f, get line: %.4f, cut img:%.4f' % (
        (toc1 - tic), (toc2 - toc1), (toc3 - toc2)
    ))

    return cut_images, (hlines, vlines)