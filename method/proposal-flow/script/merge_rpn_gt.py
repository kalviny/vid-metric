import os
import numpy as np
import glob
import scipy.io as sio

if __name__ == '__main__':

    flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/gt_flow/'
    rpn_path = '/data01/kalviny/dataset/MOT/2015/rpn/train/'
    out_path = '/data01/kalviny/dataset/MOT/2015/merge_set/merge_rpn_gt/train/'

    alt = 0

    dirs = os.listdir(flow_path)

    for v_name in dirs:
        print v_name
        v_path = os.path.join(rpn_path, v_name)
        v_list = sorted(glob.glob(os.path.join(v_path, '*.mat')))
        v_dir = os.path.join(out_path, v_name)
        print v_dir
        if not os.path.exists(v_dir): os.mkdir(v_dir)

        for idx in range(len(v_list) - 1):
            of_name = v_list[idx].split('/')[-1]
            of_bbox = []
            if os.path.exists(os.path.join(flow_path, v_name, of_name)): of_bbox = sio.loadmat(os.path.join(flow_path, v_name, of_name))['boxes']
            rpn_fn = v_list[idx + 1]
            rpn_bbox = sio.loadmat(os.path.join(rpn_fn))['boxes']
            mg_bbox = rpn_bbox
            n = len(of_bbox)
            if n: 
                mg_bbox[-n:, :] = of_bbox[0:n, :]
                mg_bbox = mg_bbox.reshape(-1, 4)
            sio.savemat(os.path.join(v_dir, rpn_fn.split('/')[-1]), {'boxes': np.array(mg_bbox, dtype=np.float32)})

        rpn_fn = v_list[0].split('/')[-1]
        rpn_bbox = sio.loadmat(os.path.join(rpn_path, v_name, rpn_fn))['boxes']
        sio.savemat(os.path.join(v_dir, rpn_fn), {'boxes': rpn_bbox.reshape(-1, 4)})

