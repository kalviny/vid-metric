import os
import numpy as np
import scipy.io as sio
import json

if __name__ == '__main__':
    in_path = '/data01/kalviny/dataset/MOT/2015/imglist/'
    out_path = '/data01/kalviny/dataset/MOT/2015/gt_box/'
    with open(os.path.join(in_path, 'train.json'), 'r') as f:
        data = json.load(f)
        for item in data:
            v_name = item['img'].split('_')[0]
            im_name = item['img'].split('_')[1]
            bbox = item['gt']
            bbox = np.array(bbox, dtype=np.float32)
            if (len(bbox)): bbox = bbox[:, 0:4]
            out_dir = os.path.join(out_path, v_name)
            if not os.path.exists(out_dir): os.mkdir(out_dir)
            sio.savemat(os.path.join(out_dir, '{}.mat'.format(im_name)), {'boxes': bbox})
            
