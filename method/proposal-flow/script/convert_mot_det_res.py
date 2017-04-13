import os
import scipy.io as sio
import numpy as np
import glob
from itertools import groupby

if __name__ == '__main__':
    vid_dir = '/data01/kalviny/dataset/MOT/2015/train/'
    #res_dir = '/home/kalviny/workspace/video_detection/video-detection/detection/faster_rcnn_vid/mx-faster-rcnn/data/MOT/2015/results/'
    res_dir = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_acf_2015_without_nms/'

    out_dir = '/data01/kalviny/dataset/MOT/2015/detection/mot_model0_2015_proposal_300/mot_acf_2015_without_nms//'
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    vid = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
            'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']

    for vid_name in vid:
        print vid_name
        im_list = sorted(os.listdir(os.path.join(vid_dir, vid_name, 'img1')))
        im_list = [str(int(x.split('.')[0])) for x in im_list]
        key = []
        val = []
        now_res_dir = os.path.join(out_dir, vid_name)
        if not os.path.exists(now_res_dir): os.mkdir(now_res_dir)

        data = np.loadtxt(os.path.join(res_dir, vid_name, 'det.txt'), delimiter=',')
        for k, v in groupby(data, lambda x: x[0]):
            key.append(str(int(k)))
            val.append(list(v))

        for i in range(len(im_list)):
            im_name = im_list[i]
            boxes = []
            if im_name in key:
                idx = key.index(im_name)
                sub_bbox = np.array(val[idx])
                sub_bbox[:, 2:4] -= 1.
                sub_bbox[:, 4:6] += sub_bbox[:, 2:4]
                boxes = np.array(sub_bbox[:, 2:6], dtype=np.float32)
                zs = np.array(sub_bbox[:, 6], dtype=np.float32)
            im_name = '%06d' % int(im_name)
            print im_name
            sio.savemat(os.path.join(now_res_dir, '{}.mat'.format(im_name)), {'boxes': boxes.reshape(-1, 4), 'zs': zs.reshape(-1, 1)})


