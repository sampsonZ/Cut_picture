import os
import cv2
import numpy as np
import glob
import time
import logging
from scipy.spatial import distance as dist

def corner_cut_shadow(img_bin):
    # _, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    con_area = [cv2.contourArea(con) for con in contours]
    filter_index = np.where(np.array(con_area) > 1000)[0]
    filter_contour = contours[filter_index[0]]
    for index in filter_index[1:]:
        filter_contour = np.vstack((filter_contour, contours[index]))
    rect = cv2.minAreaRect(filter_contour)
    box = cv2.boxPoints(rect)
    return box

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    return np.array([tl, tr, br, bl], dtype="float32")

def calculate_dist(dots):
    left_h = np.sqrt(np.square(dots[0][0]-dots[3][0]) + np.square(dots[0][1]-dots[3][1]))
    right_h = np.sqrt(np.square(dots[1][0]-dots[2][0]) + np.square(dots[1][1]-dots[2][1]))
    left_w = np.sqrt(np.square(dots[0][0]-dots[1][0]) + np.square(dots[0][1]-dots[1][1]))
    right_w = np.sqrt(np.square(dots[3][0]-dots[2][0]) + np.square(dots[3][1]-dots[2][1]))
    return (int(0.5 * (left_h + right_h)), int(0.5 * (left_w + right_w)))

def perspective_transform_one_image(img, dots, h, w):
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(dots.astype(np.float32), dst)
    img_trans = cv2.warpPerspective(img, M, (w, h))
    return img_trans, M

def get_precut(img, cols):
    h, w = img.shape[:2]
    img_gray = img.copy()
    if len(img_gray) > 2:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    img_down = cv2.pyrDown(cv2.pyrDown(img_gray))
    thre1, img_bin1 = cv2.threshold(img_down, 0, 255, cv2.THRESH_OTSU)
    box_temp = np.array([[60, 72], [5940, 91], [5939, 555], [58, 536]])
    box_temp2 = np.array([[59, 72], [5939, 57], [5941, 522], [60, 537]])
    
    bin_ratio = (np.sum(img_bin1) / 255) / (img_down.shape[0] * img_down.shape[1])
    if bin_ratio > 0.5:
        box = corner_cut_shadow(img_bin1)
        box = order_points(box) * 4
        tl, tr, br, bl = box
        per_length = calculate_dist(box)
        axis_y_status = (np.max([tl[1], tr[1]]) <= h * 0.18) and (np.min([bl[1], br[1]]) >= h * 0.83)
        axis_x_status = (np.max([tl[0], bl[0]]) <= w * 0.015) and (np.min([tr[0], br[0]]) >= w * 0.985)
        
        if axis_x_status and axis_y_status:
            img_cut, M = perspective_transform_one_image(img, box, per_length[0], per_length[1])
            
        # elif axis_y_status and abs(abs(bl[0] - tl[0]) - abs(br[0] - tr[0])) < 20 and np.hstack((img_gray[:, :int(tl[0])], img_gray[:, int(tr[0]):])).mean() < 20:
        elif abs(abs(bl[0] - tl[0]) - abs(br[0] - tr[0])) < 20 and np.hstack((img_gray[:, :int(tl[0])], img_gray[:, int(tr[0]):])).mean() < 20:
            logging.exception('find short heipian')
            cols = int(np.round((tr[0] - tl[0]) / ((w - 120) / cols * 1.0)))
            img_cut, M = perspective_transform_one_image(img, box, per_length[0], per_length[1])
            
        else:
            logging.exception('cut shadow failed, cut it by temp.')
            img_cut, M = perspective_transform_one_image(img, box_temp2, calculate_dist(box_temp2)[0], calculate_dist(box_temp2)[1])
    else:
        logging.exception('cut shadow failed, cut it by temp.')
        img_cut, M = perspective_transform_one_image(img, box_temp2, calculate_dist(box_temp2)[0], calculate_dist(box_temp2)[1])
    return img_cut, M, cols

def get_cut_line_none_precut(img, rows, cols):
    img_gray = img[:,:,0]
    hh, ww = img.shape[:2]
    xstep = ww//cols
    ystep = hh//rows
    horizontal = np.sum(img_gray, axis = 0)

    vertical = np.sum(img_gray, axis = 1)
    # 电池片水平栅格线有可能干扰到电池片的水平切割，特别处理下
    conv_width_half = hh//(rows*90)
    vertical = np.convolve(vertical, np.ones(conv_width_half*2 +1).astype('int'))[conv_width_half : -1*conv_width_half]

    col_lines = [0]
    xstart = xstep*3//4
    for x in range(cols-1):
        xstart = np.argmin(horizontal[xstart: xstart + xstep//2]) + xstart
        col_lines.append(xstart)
        xstart += xstep*3//4
    col_lines.append(ww-1)

    row_lines = [0]
    ystart = ystep*3//4
    for y in range(rows-1):
        ystart = np.argmin(vertical[ystart: ystart + ystep//2]) + ystart
        row_lines.append(ystart)
        ystart += ystep*3//4
    row_lines.append(hh-1)

    # verify
    assert abs(col_lines[-1]-col_lines[-2]-xstep)/xstep < 0.4
    assert abs(row_lines[-1]-row_lines[-2]-ystep)/ystep < 0.4

    return col_lines, row_lines
    
def grid_cut(img_data, cols=27):
    try:
        img_data, M, cols = get_precut(img_data, cols)
        col_lines, row_lines = get_cut_line_none_precut(img_data, 1, cols)
    except:   
        logging.exception('el cut failed')
        return None, None, None, None
    cut_images = {}
    new_col_lines = []
    if cols % 2 == 1:
        for col in np.arange(0, len(col_lines)-2, 2):
            key = (0, col_lines[col])
            img = img_data[:, col_lines[col]:col_lines[col+2], :]
            cut_images[key] = img
            new_col_lines.append(col_lines[col])
        last_img = img_data[:, col_lines[-2]:col_lines[-1], :]
        cut_images[(0, col_lines[-2])] = np.hstack((last_img, np.zeros(last_img.shape).astype(int)))
        new_col_lines.append(col_lines[-2])
        new_col_lines.append(col_lines[-2] + 2 * last_img.shape[1])
    else:
        for col in np.arange(0, len(col_lines)-1, 2):
            key = (0, col_lines[col])
            img = img_data[:, col_lines[col]:col_lines[col+2], :]
            cut_images[key] = img
            new_col_lines.append(col_lines[col])
        new_col_lines.append(col_lines[-1])
    return cut_images, row_lines, new_col_lines, M