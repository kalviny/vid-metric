import os
import scipy.io as sio
import numpy as np
import glob
from itertools import groupby

if __name__ == '__main__':
    vid_dir = '/data01/kalviny/dataset/kitti/training/image_02/'

    res_dir = '/home/kalviny/workspace/experiment/mx-faster-rcnn-kitti/result/kitti_proposal_300/kitti_acf_weighted_nms_pf/'

    out_dir = '/data01/kalviny/dataset/kitti/detection/kitti_proposal_300/car/kitti_acf_weighted_nms_pf/'

    cls = 'Car'
    vid = ['0011', '0012', '0013', '0014', '0015', '0016', '0018', '0009', '0020']
    #cls = 'Pedestrian'
    #vid = ['0011', '0012', '0013', '0014', '0015', '0017', '0016', '0009']

    if not os.path.exists(out_dir): os.mkdir(out_dir)


    for vid_name in vid:
        print vid_name
        im_list = sorted(os.listdir(os.path.join(vid_dir, vid_name)))
        im_list = [str(int(x.split('.')[0])) for x in im_list]
        key = []
        val = []
        now_res_dir = os.path.join(out_dir, vid_name)
        if not os.path.exists(now_res_dir): os.mkdir(now_res_dir)

        data = np.loadtxt(os.path.join(res_dir, vid_name, '{}.txt'.format(cls)), delimiter=' ')
        for k, v in groupby(data, lambda x: x[0]):
            key.append(str(int(k)))
            val.append(list(v))

        for i in range(len(im_list)):
            im_name = im_list[i]
            boxes = []
            zs = []
            if im_name in key:
                idx = key.index(im_name)
                sub_bbox = np.array(val[idx])

                boxes = np.array(sub_bbox[:, 2:], dtype=np.float32)

                zs = np.array(sub_bbox[:, 1], dtype=np.float32)

                sio.savemat(os.path.join(now_res_dir, '{}.mat'.format('%06d' % int(im_name))), {'boxes': boxes.reshape(-1, 4), 'zs': zs.reshape(-1, 1)})
            else:
                sio.savemat(os.path.join(now_res_dir, '{}.mat'.format('%06d' % int(im_name))), {'boxes': np.array(boxes), 'zs': np.array(zs)})

            im_name = '%06d' % int(im_name)
            print im_name


