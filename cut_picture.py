import cv2
import os
import fire
import glob
from utils.cut_zujian_pic import grid_cut as grid_cut_tm


def main(old_path, save_path, rows=6, cols=10, half_plate=False, edge_removed=False):
    s_pic_file = glob.glob(os.path.join(old_path, 'img', '*.jpg'))

    pic_file = s_pic_file[0]

    img_data = cv2.imread(pic_file, 0)
    cut_images, (row_lines, col_lines) = grid_cut_tm(img_data, rows, cols, edge_removed=edge_removed)

    for key in cut_images.keys():
        a = 1
        cv2.imwrite(os.path.join(save_path, 'img', '%s.jpg' % a),
                    key)
        a += 1


if __name__ == '__main__':
    main(old_path='D:\Data\el', save_path='D:\Data\el_cut')
