import os
import numpy as np
import json
from itertools import groupby
from prettytable import PrettyTable
from trajectory.eval_assn import EvalAssn
from mAP.eval_map import EvalmAP
import math
import copy

cls = [1]
#thr = np.arange(0, 1.0 + 0.1, 0.1)
#thr[0] += 1e-3
thr = [0.5]

def get_auc(x, y):
    x = np.array(x)
    y = np.array(y)

    x = np.concatenate(([0.], x))
    y = np.concatenate(([0.], y))

    i = np.where(y[1:] != y[:-1])[0]

    ap = np.sum((x[i + 1] - x[i]) * y[i + 1])

    return ap

def process_ap(x, y):
    y = np.concatenate(([0.],y, [0.]))
    x = np.concatenate(([0.], x, [1.]))
    for i in range(y.size - 1, 0, -1):
        y[i - 1] = np.maximum(y[i - 1], y[i]) 
    rec = []
    prec = []
    for i in range(len(x)):
        rec.append(str(x[i]))
    for i in range(len(y)):
        prec.append(str(y[i]))
    return rec, prec


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
    vid_path = '/data01/kalviny/dataset/MOT/2015/train/'

    method = 'mot_model0_2015_nms_0.5'
    #method = 'mot_model0_2015_nms_0.5_mf'

    #method = 'mot_model0_2015_weighted_nms'
    #method = 'mot_model0_2015_weighted_nms_mf'

    #method = 'mot_model0_2015_pf_window_5_nms'
    #method = 'mot_model0_2015_pf_window_5_weighted_nms'

    #method = 'mot_model0_2015_pf_window_3_nms'
    #method = 'mot_model0_2015_pf_window_3_weighted_nms'

    #method = 'mot_model0_2015_pf_window_7_nms'
    #method = 'mot_model0_2015_pf_window_7_weighted_nms'

    #method = 'mot_model0_2015_nms_0.5_mf_pf'
    #method = 'mot_model0_2015_weighted_nms_mf_pf'

    #method = 'mot_model0_2015_without_nms/'
    #method = 'mot_model0_2015_nms_0.5_mf'

    # ---------------------- #
    #method = 'mot_acf_2015_weighted_nms_mf'
    #method = 'mot_acf_2015_weighted_nms'
    #method = 'mot_acf_2015_nms'
    #method = 'mot_acf_2015_nms_mf'
    #method = 'mot_acf_2015_nms_pf'
    #method = 'mot_acf_2015_weighted_nms_pf'
    #method = 'mot_acf_2015_nms_mf_pf'
    #method = 'mot_acf_2015_weighted_nms_mf_pf'

    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']

    det_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/' + method
    # ------ acf path --------
    #$det_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_acf_2015_proposal_300/' + method

    #method = 'mot_acf_2015_pf_window_3_weighted_nms'
    #det_path = '/home/kalviny/workspace/video_detection/tmp/video-detection/metric/method/proposal-flow/script/result/'

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
                #if (int(x[7]) in [1, 7]) and x[8] >= visual_thr:
                    #filter_seq.append(x)
                filter_seq.append(x)

            seq_gt = np.array(filter_seq)
            seq_gt[:, 2:4] -= 1.
            seq_gt[:, 4:6] += seq_gt[:, 2:4]

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
            #seq_det = np.loadtxt(os.path.join(det_path, vid_name, 'seq_nms.txt'), delimiter=',')
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
                #_bbox = val[x][:, 2:6]
                _bbox = val[x][:, 2:7]
                det_traj[im_name] = {'bbox': np.array(_bbox)}


            st_out_dir = 'result/mot2015/stability/model0/' + method
            if not os.path.exists(st_out_dir): os.mkdir(st_out_dir)

            out_dir = 'result/mot2015/accuracy/model0/' + method
            if not os.path.exists(out_dir): os.mkdir(out_dir)

            fout_st = open(os.path.join(st_out_dir, '{}.txt'.format(vid_name)), 'w')
            with open(os.path.join(out_dir, '{}.txt'.format(vid_name)), 'w') as f:
                for th in thr:
                    _det_map = copy.deepcopy(det_map)
                    _anno = copy.deepcopy(anno)
                    _rec, _prec, _ap, _rec_ls = EvalmAP(_anno, np.array(_det_map), th)

                    frg = []
                    ce = []
                    re = []
                    x = []
                    sc = _rec_ls[-2][1]
                    print sc
                    _, _, _ = EvalAssn(anno, det_traj, vid_name, sc, th)
                    '''
                    for item in _rec_ls:
                        sc_thr = item[1]
                        x.append(item[0])
                        _err, _c, _r = EvalAssn(anno, det_traj, sc_thr, th)
                        if math.isnan(_err): _err = 0
                        if math.isnan(_c): _c = 0
                        if math.isnan(_r): _r = 0
                        frg.append(_err)
                        ce.append(_c)
                        re.append(_r)

                    frg_auc = get_auc(x, frg)
                    ce_auc = get_auc(x, ce)
                    re_auc = get_auc(x, re)
                    fout_st.write('%.2f, %.4f, %.4f, %.4f\n' % (th, frg_auc,  ce_auc,  re_auc))

                    if math.isnan(_ap): _ap = 0
                    if th == 0.5:
                        ap = _ap
                        F_err = frg_auc
                        var_c = ce_auc
                        var_r = re_auc

                        seq_res.append(np.array([ap, F_err, var_c, var_r]))
                        tab.add_row([vid_name, ap, F_err, var_c, var_r])

                    f.write('%.2f, %.2f\n' % (th, _ap))
                    print th, _ap, frg_auc, ce_auc, re_auc
            fout_st.close()

            
        seq_res = np.array(seq_res)
        tab.add_row(['----------', '-----------', '----------', '-----------', '----------'])
        tab.add_row(['Mean', np.mean(seq_res[:, 0]), np.mean(seq_res[:, 1]), np.mean(seq_res[:, 2]), np.mean(seq_res[:, 3])])
    print tab
    '''

