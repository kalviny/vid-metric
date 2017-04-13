import os
import numpy as np
import json
from itertools import groupby
from prettytable import PrettyTable
from trajectory.eval_assn import EvalAssn
from mAP.eval_map import EvalmAP

cls = [1]
thr = np.arange(0, 1.0 + 0.05, 0.05)
thr = [0.5]

def save_fig(path, vid_name, x, y):
    if not os.path.exists(path): os.mkdir(path)
    import matplotlib.pyplot as plt
    y = np.concatenate(([0.],y, [0.]))
    x = np.concatenate(([0.], x, [1.]))
    for i in range(y.size - 1, 0, -1):
        y[i - 1] = np.maximum(y[i - 1], y[i]) 

    plt.figure(0)
    plt.plot(x, y)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('%s' % vid_name)
    plt.savefig(os.path.join(path, '{}.png'.format(vid_name)))
    plt.close(0)


def get_im_name(im_name):
    return str(int(im_name))

def conv_bbox(bbox):    
    b = [float(x) for x in bbox]
    sub = []
    x1 = b[0] - 1.
    y1 = b[1] - 1.
    x2 = x1 + b[2]
    y2 = y1 + b[3]
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    vid_path = '/data01/kalviny/dataset/MOT/2016/train/'
    vid_list = os.listdir(os.path.join(vid_path))
    vid_list = sorted(vid_list, key = lambda x: int(x.split('-')[1]))
    visual_thr = 0.2
    method = 'mot_model2_nms_change_rpn'
    #method = 'mot_model2_weighted_nms_change_rpn_0.5_0.7'

    #det_path = '/home/yhli/mx-faster-rcnn/result/mot_model2/'
    det_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/' + method

    # range(len(vid_list))

    tab = PrettyTable(['video', 'ap@0.5', 'Fragment', 'Center Error', 'Ratio Error'])
    tab.align['video']
    tab.padding_width = 1
    # len(vid_list)
    for c in cls:
        seq_res = []
        for i in range(len(vid_list)):
            vid_name = vid_list[i]
            print 'processing {}'.format(vid_name)
            seq_gt = np.loadtxt(os.path.join(vid_path, vid_name, 'gt/gt.txt'), delimiter=',')
            anno = {}
            filter_seq = []
            for x in seq_gt:
                if int(x[6]) == 0: continue
                #if (int(x[7]) == c or int(x[7]) == 7) and x[8] >= visual_thr:
                if (int(x[7]) in [1, 7]) and x[8] >= visual_thr:
                    filter_seq.append(x)

            seq_gt = np.array(filter_seq)
            seq_gt[:, 2:4] -= 1.
            seq_gt[:, 4:6] += seq_gt[:, 2:4]

            #seq_gt[:, 2:4] = np.maximum(0, seq_gt[:, 2:4])

            seq_gt = sorted(seq_gt, key = lambda x: x[0])
            key = []
            val = []
            for k, v in groupby(seq_gt, key = lambda x: x[0]):
                key.append(k)
                val.append(np.array(list(v)))
            for k in range(len(key)):
                im_name = get_im_name(key[k])
                _bbox = val[k][:, 2:6]
                traj_id = val[k][:, 1]
                anno[im_name] = {'bbox': np.array(_bbox, dtype=np.float32), 'det': [False] * len(traj_id), 'traj_id': traj_id}

            det_map = []
            det_traj = {}
            seq_det = np.loadtxt(os.path.join(det_path, vid_name, 'det.txt'), delimiter=',')
            seq_det = np.array(sorted(seq_det, key = lambda x: x[0]))
            seq_det[:, 2:4] -= 1.
            seq_det[:, 4:6] += seq_det[:, 2:4]

            for x in seq_det:
                im_name = get_im_name(x[0])
                _sub = []
                _sub.append(im_name)
                _sub.append(x[6])
                _sub.extend(x[2:6])
                det_map.append(_sub)
                #det_map.append([im_name, x[6], x[2:6]])
            key = []
            val = []
            for k, v in groupby(seq_det, key = lambda x: x[0]):
                key.append(k)
                val.append(np.array(list(v)))
            for x in range(len(key)):
                im_name = get_im_name(key[x])
                _bbox = val[x][:, 2:6]
                det_traj[im_name] = {'bbox': np.array(_bbox)}

            #_err, _c, _r = EvalAssn(anno, det_traj, vid_name, 0.9)
            #rec, prec, ap = EvalmAP(anno, np.array(det_map))


            out_dir = 'result/mot2016/traj/' + method
            if not os.path.exists(out_dir): os.mkdir(out_dir)

            with open(os.path.join(out_dir, '{}.txt'.format(vid_name)), 'w') as f:
                for th in thr:
                    _err, _c, _r = EvalAssn(anno, det_traj, th)
                    if th == 0.5:
                        F_err = _err
                        var_c = _c
                        var_r = _r
                    print th, _err, _c, _r
                    f.write('%.2f, %.2f, %.2f, %.2f\n' % (th, _err, _c, _r))

            rec, prec, ap = EvalmAP(anno, np.array(det_map))
            seq_res.append(np.array([ap, F_err, var_c, var_r]))

            out_dir = 'result/mot2016/mAP/' + method
            #out_dir = 'result/mot2016/mAP/weighted_nms_0.3_0.7/'
            if not os.path.exists(out_dir): os.mkdir(out_dir)

            save_fig(out_dir, vid_name, rec, prec)

            tab.add_row([vid_name, ap, F_err, var_c, var_r])

        seq_res = np.array(seq_res)
        tab.add_row(['----------', '-----------', '----------', '-----------', '----------'])
        tab.add_row(['Mean', np.mean(seq_res[:, 0]), np.mean(seq_res[:, 1]), np.mean(seq_res[:, 2]), np.mean(seq_res[:, 3])])
    print tab


