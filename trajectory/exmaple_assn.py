import os
import numpy as np
from itertools import groupby
from eval_assn import EvalAssn

def ConvBbox(bbox):
    traj_id = []
    res_box = []
    for i in range(len(bbox)):
        b = bbox[i].strip().split(',')
        new_bbox = []
        if (float(b[6]) == 0): continue
        new_bbox.append(float(b[2]) - 1.)
        new_bbox.append(float(b[3]) - 1.)
        new_bbox.append(float(b[2]) + float(b[4]) - 1.)
        new_bbox.append(float(b[3]) + float(b[5]) - 1.)
        res_box.append(new_bbox)
        traj_id.append(b[1])
    return traj_id, res_box

if __name__ == '__main__':
    img = '/data01/kalviny/dataset/MOT/2015/train/ADL-Rundle-6/img1/'
    im_list = os.listdir(img)
    im_list = [x.split('.')[0] for x in im_list]
    im_list = sorted(im_list)

    gt_file = '/data01/kalviny/dataset/MOT/2015/train/ADL-Rundle-6/gt/gt.txt'

    anno = {}

    key = []
    val = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for k, v in groupby(lines, lambda x: x.strip('\n').split(',')[0]):
            key.append(k)
            val.append(list(v))
    for i in range(len(key)):
        _im_list = im_list[int(key[i]) - 1]
        bbox = []
        traj_id = []
        traj_id, bbox = ConvBbox(val[i])
        _val = {'bbox': np.array(bbox, dtype=np.float32), "traj_id": traj_id}
        anno[_im_list] = _val

    det_file = '/home/kalviny/workspace/video_detection/video-detection/detection/faster_rcnn_vid/mx-faster-rcnn/data/MOT/2015/results/ADL-Rundle-6/ADL-Rundle-6.txt'

    det = {}
    key = []
    val = []
    with open(det_file, 'r') as f:
        lines = f.readlines()
        for k, v in groupby(lines, lambda x: x.strip().split(',')[0]):
            key.append(k.split('.')[0])
            val.append(list(v))

        for i in range(len(key)):
            bbox = []
            for j in range(len(val[i])):
                sub_bbox = val[i][j].split(',')[1:5]
                print '-------------------------'
                print sub_bbox
                print '-------------------------'
                bbox.append(sub_bbox)

            _val = {'bbox': np.array(bbox, dtype=np.float32)}
            det[key[i]] = _val

    f, c, r = EvalAssn(anno, det)
    print f, c, r


