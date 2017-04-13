import os
import numpy as np
import glob
import scipy.io as sio

if __name__ == '__main__':

    in_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mat/'
    rpn_path = '/data01/kalviny/dataset/MOT/2015/rpn/train/'
    out_path = '/data01/kalviny/dataset/MOT/2015/merge_rpn_of/train/'

    dirs = os.listdir(in_path)
    for v_name in dirs:
        print v_name
        v_path = os.path.join(in_path, v_name)
        v_list = sorted(glob.glob(os.path.join(v_path, '*.mat')))
        v_dir = os.path.join(out_path, v_name)
        print v_dir
        if not os.path.exists(v_dir): os.mkdir(v_dir)

        for idx in range(len(v_list) - 1):
            of_bbox = sio.loadmat(v_list[idx])['boxes']
            rpn_fn = v_list[idx + 1].split('/')[-1]
            rpn_bbox = sio.loadmat(os.path.join(rpn_path, v_name, rpn_fn))['boxes']
            mg_bbox = np.concatenate((rpn_bbox, of_bbox))
            sio.savemat(os.path.join(v_dir, rpn_fn), {'boxes': mg_bbox.reshape(-1, 4)})

        rpn_fn = v_list[0].split('/')[-1]
        rpn_bbox = sio.loadmat(os.path.join(rpn_path, v_name, rpn_fn))['boxes']
        sio.savemat(os.path.join(v_dir, rpn_fn), {'boxes': rpn_bbox.reshape(-1, 4)})

