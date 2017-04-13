import os
import cv2
from itertools import groupby
import numpy as np

def get_start(keep_now):
    res = -1
    for idx in keep_now:
        if len(keep_now[idx]): 
            res = idx
            break
    return res

def get_iou(a, b):
    #print len(a), len(b)
    #print a.shape, b.shape

    b = b.reshape(-1, 4)

    if (not len(a)) or (not len(b)): return 0

    x_min = np.maximum(a[0], b[:, 0])
    y_min = np.maximum(a[1], b[:, 1])
    x_max = np.minimum(a[2], b[:, 2])
    y_max = np.minimum(a[3], b[:, 3])

    w = np.maximum(0, x_max - x_min + 1.)
    h = np.maximum(0, y_max - y_min + 1.)

    inter = w * h
    uni = (a[2] - a[0] + 1.) * (a[3] - a[1] + 1.) + (b[:, 2] - b[:, 0] + 1.) * (b[:, 3] - b[:, 1] + 1.) - inter
    ov = inter / uni

    return ov


def find_best_seq(proposal, st, thr=0.5):
    best_seq = []
    rescore = 0
    #print proposal[st]
    idx = np.argmax(proposal[st][:, 4])
    best_seq.append(idx)
    now_bbox = proposal[st][idx, 0:4]
    rescore += proposal[st][idx, 4]
    for i in range(st + 1, len(proposal)):
        ov = get_iou(now_bbox, proposal[i][:, 0:4])
        candidate_bbox = np.where(ov >= thr)[0]
        if not len(candidate_bbox): break
        idx = np.argmax(proposal[i][candidate_bbox, 4])
        idx = candidate_bbox[idx]
        best_seq.append(idx)
        print '--------->>', now_bbox, '<<--------->>', proposal[i][idx, 0:4], '<<-----------', get_iou(now_bbox, proposal[i][idx, 0:4]), '--------------'
        now_bbox = proposal[i][idx, 0:4]
        #rescore = max(rescore, proposal[i][idx, 4])
        rescore += proposal[i][idx, 4]
        #if (len(best_seq) >= 7): break

    # return average score
    #print rescore * 1. / len(best_seq)
    return best_seq, rescore * 1. / len(best_seq)

def seq_nms(proposal, nms_thr=0.5):
    res_seq_nms = {}
    for i in range(len(proposal)): res_seq_nms[i] = []
    while True:
        st = int(get_start(proposal))
        if st == -1: return res_seq_nms
        best_seq, rescore = find_best_seq(proposal, st, 0.5)
        for i in range(len(best_seq)):
            ov = get_iou(proposal[st + i][best_seq[i], 0:4], proposal[st + i][:, 0:4])
            _sub = proposal[st + i][best_seq[i]]
            _sub[4] = rescore
            res_seq_nms[st + i].append(_sub)
            new_idx = np.where(ov < nms_thr)[0]

            proposal[st + i] = proposal[st + i][new_idx, :]

    return res_seq_nms

if __name__ == '__main__':
    vid_res = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_model0_2015_without_nms'
    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']

    vid_list = ['PETS09-S2L1']

    out_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_model0_2015_without_nms'

    # add bonus
    #thresh = 1e-3

    for vid_name in vid_list:

        seq_det = np.loadtxt(os.path.join(vid_res, vid_name, 'det.txt'), delimiter = ',')
        #seq_det = np.loadtxt('det.txt', delimiter = ',')
        seq_det[:, 2:4] -= 1.
        seq_det[:, 4:6] += seq_det[:, 2:4]

        key = []
        val = []
        for k, v in groupby(seq_det, key = lambda x: x[0]):
            key.append(k)
            val.append(list(v))

        print 'processing %s, %d images' % (vid_name, len(key))

        proposal = {}
        for j in range(len(key)):
            _val = np.array(sorted(val[j], key = lambda x: x[6])[::-1])
            #_val = _val[np.where(_val[7] >= thresh)[0]
            proposal[j] = np.array(_val[:, 2:7])

        res_seq_nms = seq_nms(proposal)

        #with open(os.path.join(out_path, vid_name, 'det_seq.txt'), 'w') as f:
        ratio = 0.1
        bonus = 4e-3
        #with open('det_seq.txt', 'w') as f:
        with open(os.path.join(out_path, vid_name, 'det_seq.txt'), 'w') as f:
            for fr_idx in res_seq_nms:
                ls = np.array(res_seq_nms[fr_idx])
                ls[:, 2:4] -= ls[:, 0:2]
                ls[:, 0:2] += 1.0
                ls = np.array(sorted(ls, key = lambda x: x[4])[::-1])
                top_cls = int(np.floor(len(ls) * ratio))
                if top_cls > 0: ls[0:top_cls, 4] += bonus
                for i in range(len(ls)):
                    f.write('%d,-1,%.6f,%.6f,%.6f,%.6f,%.6f,1,-1,-1\n' % (fr_idx + 1, ls[i, 0], ls[i, 1], ls[i, 2], ls[i, 3], ls[i, 4]))
