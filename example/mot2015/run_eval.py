import os
import numpy as np
import pylab as pl
import json
from itertools import groupby
from prettytable import PrettyTable
from trajectory.eval_assn import EvalAssn
from mAP.eval_map import EvalmAP

cls = ['1']

def get_im_name(im_name):
    return str(int(im_name))

def conv_bbox(bbox):    
    b = [float(x) for x in bbox]
    sub = []
    sub.append(b[0])
    sub.append(b[1])
    sub.append(b[0] + b[2] - 1.)
    sub.append(b[1] + b[3] - 1.)
    return sub


if __name__ == '__main__':
    vid_path = '/data01/kalviny/dataset/MOT/2016/train/'
    vid_list = os.listdir(os.path.join(vid_path))
    vid_list = sorted(vid_list, key = lambda x: int(x.split('-')[1]))

    det_path = '/home/kalviny/workspace/video_detection/video-detection/detection/faster_rcnn_vid/mx-faster-rcnn/data/MOT/2015/results/'

    tab = PrettyTable(['video', 'ap', 'Fragment', 'Center Error', 'Ratio Error'])
    tab.align['video']
    tab.padding_width = 1
    # len(vid_list)
    for i in range(len(vid_list)):
        vid_name = vid_list[i]
        im_list = os.listdir(os.path.join(vid_path, vid_name, 'img1'))
        im_list = [x.split('.')[0] for x in im_list]
        print 'processing {}'.format(vid_name)
        for c in cls:
            anno = {}
            with open(os.path.join(vid_path, vid_name, 'gt/gt.json'), 'r') as f:
                gt = json.load(f)
                for j in range(len(gt)):
                    _gt = gt[j]
                    im_name = get_im_name(_gt['img'])

                    _traj = _gt['traj_id']
                    _cls = np.array(_gt['cls'])
                    _bbox = _gt['bbox']
                    _bbox = np.array(_gt['bbox'], dtype=np.float32)

                    tmp = len(_bbox)
                    _cls = np.where(_cls == c)

                    _bbox = [_bbox[x] for x in _cls[0]]
                    _traj = [_traj[x] for x in _cls[0]]

                    anno[im_name] = {'bbox': np.array(_bbox, dtype=np.float32), 'det': [False] * len(_cls[0]), 'traj_id': _traj}
            det_map = []
            det_traj = {}
            with open(os.path.join(det_path, vid_name, '{}.txt'.format(vid_name)), 'r') as f:
                lines = f.readlines()
                lines = [x.strip() for x in lines]
                lines = sorted(lines, key=lambda x: int(x.split(',')[0]))
                for x in lines:
                    x = x.split(',')
                    im_name = get_im_name(x[0])
                    sub = []
                    sub.append(im_name)
                    sub.append(x[5])
                    sub.extend(conv_bbox(x[1:5]))
                    det_map.append(sub)
                key = []
                val = []
                for k, v in groupby(lines, key=lambda x: x.split(',')[0]):
                    key.append(k)
                    val.append(list(v))
                for x in range(len(key)):
                    _bbox = []
                    for v in val[x]:
                        _bbox.append(conv_bbox(v.split(',')[1:5]))
                    im_name = get_im_name(key[x])
                    det_traj[im_name] = {'bbox': np.array(_bbox)}

            F_err, var_c, var_r = EvalAssn(anno, det_traj)
            rec, prec, ap = EvalmAP(anno, np.array(det_map))
            pl.plot(rec, prec)
            pl.show()
            tab.add_row([vid_name, ap, F_err, var_c, var_r])
    print tab


