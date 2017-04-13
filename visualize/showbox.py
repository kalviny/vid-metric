import cv2
import numpy as np
from itertools import groupby
import os

if __name__ == '__main__':
    det_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/'
    #method = 'mot_model0_2015_nms_0.5'
    method = 'mot_model0_2015_rescore_nms'
    path = det_path + method

    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']
    vid_list = ['KITTI-13']

    img_path = '/data01/kalviny/dataset/MOT/2015/train/'

    for vid_name in vid_list:
        with open(os.path.join(path, vid_name, 'det.txt'), 'r') as f:
            data = np.loadtxt(f, delimiter = ',')

            val = []
            key = []

            for k, v in groupby(data, key = lambda x: x[0]):
                key.append(k)
                val.append(np.array(list(v)))

            for idx, im_name in enumerate(key):
                im = cv2.imread(os.path.join(img_path, vid_name, 'img1', '{}.jpg'.format('%06d' % im_name)))
                _val = val[idx]
                for i in range(len(_val)):
                    bbox = _val[i, :]
                    if bbox[5] < 0.5: continue
                    cv2.rectangle(im, (int(bbox[2]) - 1, int(bbox[3]) - 1), (int(bbox[2]) + int(bbox[4]) - 1, int(bbox[3]) + int(bbox[5]) - 1), (30, 144, 255), 4)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(im, '%.2f' % bbox[6], (int(bbox[2]) - 1, int(bbox[3]) - 1), font, 1, (255, 0, 0), 2)
                out_dir = 'output/' + method
                if not os.path.exists(out_dir): os.mkdir(out_dir)
                cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format('%06d' % im_name)), im)




