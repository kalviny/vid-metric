import cv2
import numpy as np
import os
import json
import colorsys

def get_color():
    h = np.random.uniform(0.02, 0.31) + np.random.choice([0, 1./3, 2./3])
    l = np.random.uniform(0.3, 0.8)
    s = np.random.uniform(0.3, 0.8)
    rgb = colorsys.hls_to_rgb(h, l, s)
    return (int(rgb[0] * 256), int(rgb[1] * 256), int(rgb[2] * 256))


if __name__ == '__main__':
    vid_path = '/data01/kalviny/dataset/MOT/2015/train/'
    #vid_list = os.listdir(os.path.join(vid_path))
    #vid_list = sorted(vid_list, key = lambda x: int(x.split('-')[1]))
    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']


    method = 'mot_model0_2015_with_tr_weighted_nms'

    for i in range(len(vid_list)):
        fourcc = cv2.cv.FOURCC(*"DIVX")
        print vid_list[i]
        col = {}
        vid_name = vid_list[i]
        im = cv2.imread(os.path.join(vid_path, vid_name, 'img1', '000001.jpg'))
        h, w, _ = im.shape
        if not os.path.exists(method): os.mkdir(method)
        out = cv2.VideoWriter(os.path.join(method, '{}.avi'.format(vid_name)), fourcc, 15, (w, h))
        p = '2015/output/' + method 
        im_list = os.listdir(os.path.join(vid_path, vid_name, 'img1'))
        im_list = sorted(im_list, key = lambda x: int(x.split('.')[0]))
        with open(os.path.join(p, '{}.json'.format(vid_name)), 'r') as f:
            data = json.load(f)
            for j in range(len(im_list)):
                print im_list[j]
                im = cv2.imread(os.path.join(vid_path, vid_name, 'img1', im_list[j]))
                im_name = str(int(im_list[j].split('.')[0]))
                if im_name in data:
                    ls = data[im_name]
                    for x in range(len(ls)):
                        b = ls[x]
                        if not b[0] in col.keys(): col[b[0]] = get_color()
                        sub_col = col[b[0]]
                        cv2.rectangle(im, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])), sub_col, 4)
                    out.write(im)
                else:
                    out.write(im)

