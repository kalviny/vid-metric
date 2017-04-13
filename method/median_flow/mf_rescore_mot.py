from __future__ import print_function

import cv2
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import scipy
import scipy.io as sio
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import colorsys

iou_thr = 0.3

def iou(bb_test,bb_gt):
    """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    if (not len(bb_test)) or (not len(bb_gt)): return 0

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

class Bbox:
    def __init__(self):
        self.pos = []
        self.next_bbox = []
        self.frame = 0
        self.opt_flow = []

class Traj:
    def __init__(self):
        self.bbox = []
        self.fr = 0
        self.idx = 0

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[0:4], trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for d,det in enumerate(trackers):
        if(d not in matched_indices[:,1]):
            unmatched_trackers.append(d)

      #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class MF(object):
    def __init__(self):
        self.trackers = []

        self.count = 0
        self.traj = {}

    def update(self, dets, of_fw, of_bw, fr_name):
        '''
            dets - a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        '''

        new_dets = []
        
        trks_of = []
        for b in self.trackers:
            trks_of.append(b[-1].bbox.opt_flow)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks_of, iou_thr)

        f_trk = []
        f_det = []
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0],0]
                # BW tracking
                bw_iou = iou(of_bw[d[0], :], trk[-1].bbox.pos)
                if (dets[d[0], 4] < 0.9): 
                    f_det.append(d[0])
                    f_trk.append(t)
                    continue
                if (bw_iou <= iou_thr): 
                    f_trk.append(t)
                    f_det.append(d[0])
                    continue
                new_bbox = Bbox()
                new_bbox.pos = dets[d[0], :]
                new_bbox.opt_flow = of_fw[d[0], :]

                # avg
                x1 = (trk[-1].bbox.opt_flow[0] + dets[d[0], 0]) / 2.
                y1 = (trk[-1].bbox.opt_flow[1] + dets[d[0], 1]) / 2.
                x2 = (trk[-1].bbox.opt_flow[2] + dets[d[0], 2]) / 2.
                y2 = (trk[-1].bbox.opt_flow[3] + dets[d[0], 3]) / 2.

                #x1 = dets[d[0], 0]
                #y1 = dets[d[0], 1]
                #x2 = dets[d[0], 2]
                #y2 = dets[d[0], 3]

                new_dets.append(np.hstack((x1, y1, x2, y2, dets[d[0], 4])))

                new_traj = Traj()
                new_traj.bbox = new_bbox
                new_traj.fr = fr_name
                new_traj.idx = d[0]

                self.trackers[t].append(new_traj)

        for t in range(len(self.trackers) - 1, -1, -1):
            if (t in unmatched_trks):
                f_trk.append(t)
                if (self.trackers[t][-1].bbox.pos[4] <= 0.8):
                    f_trk.append(t)
                else:
                    trk = self.trackers[t][-1]
                    new_bbox = Bbox()
                    new_bbox.pos = np.hstack((trk.bbox.opt_flow, trk.bbox.pos[4] * 0.5))
                    new_bbox.opt_flow = trk.bbox.opt_flow
                    new_bbox.prev_box = trk.bbox.pos

                    new_traj = Traj()
                    new_traj.bbox = new_bbox
                    new_traj.fr = fr_name
                    new_traj.idx = -1

                    self.trackers[t].append(new_traj)

        f_trk = list(set(f_trk))
        for t in range(len(self.trackers) - 1, -1, -1):
            if (t in f_trk):

                tmp = []
                # append failed trajectory
                for x in range(len(self.trackers[t])):
                    b = self.trackers[t][x]
                    if b.bbox.pos[4] < 0.3: continue
                    tmp.append(b)
                #self.traj[self.count] = np.copy(self.trackers[t])
                self.traj[self.count] = tmp
                self.count = self.count + 1

                self.trackers.pop(t)

        # append new dets in active pool
        unmatched_dets = np.concatenate((unmatched_dets, f_det))
        unmatched_dets = list(set(unmatched_dets))
        unmatched_dets = np.asarray(unmatched_dets, dtype=np.uint8)
        for d in unmatched_dets:
            new_dets.append(dets[d, :])
            if (dets[d, 4] < 0.9): continue

            new_bbox = Bbox()
            new_bbox.pos = dets[d, :]
            new_bbox.opt_flow = of_fw[d, :]

            new_traj = Traj()
            new_traj.bbox = new_bbox
            new_traj.fr = fr_name
            new_traj.idx = d
            self.trackers.append([new_traj])

        return np.array(new_dets)

def display(traj, vid_name, path):
    for item in traj:
        bbox = item.bbox.pos
        print(bbox[4])
        im_name = '%06d' % (int(item.fr))
        im = cv2.imread(os.path.join(path, vid_name, 'img1', '{}.jpg'.format(im_name)))
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (30, 144, 255), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, '%.2f' % bbox[4], (int(bbox[0]), int(bbox[1])), font, 1, (255, 255, 255), 2)
        cv2.imshow('im', im)
        cv2.waitKey(2)

if __name__ == '__main__':

    det_path = '/data01/kalviny/dataset/MOT/2015/detection/mot_model0_2015_proposal_300/mot_model0_2015_nms_0.5/'

    bw_flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_model0_2015_proposal_300/mot_model0_2015_nms_0.5/bw/'
    fw_flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_model0_2015_proposal_300/mot_model0_2015_nms_0.5/fw/'

    out_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_model0_2015_rescore_nms/'

    data_path = '/data01/kalviny/dataset/MOT/2015/train/'

    #vid_list = ['TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof']
    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']

    for vid_name in vid_list:
        mot_tracker = MF()

        seq_dets = {}

        print(vid_name)
        im_list = os.listdir(os.path.join(data_path, vid_name, 'img1'))
        im_list = sorted(im_list, key = lambda x: int(x.split('.')[0]))

        if not os.path.exists(os.path.join(out_path, vid_name)): os.makedirs(os.path.join(out_path, vid_name))
        out_dir = os.path.join(out_path, vid_name)
        with open(os.path.join(out_dir, 'det.txt'), 'w') as f:
            print(len(im_list))
            for j in range(len(im_list)):
                fr_name = im_list[j].split('.')[0]
                _seq_dets = sio.loadmat(os.path.join(det_path, vid_name, '{}.mat'.format(fr_name)))
                _dets = np.hstack((_seq_dets['boxes'], _seq_dets['zs']))
                seq_dets[fr_name] = _dets

                try:
                    of_fw = sio.loadmat(os.path.join(fw_flow_path, vid_name, '{}.mat'.format(fr_name)))['boxes']
                    of_bw = sio.loadmat(os.path.join(bw_flow_path, vid_name, '{}.mat'.format(fr_name)))['boxes']
                except IOError:
                    of_fw = []
                    of_bw = []

                new_dets = mot_tracker.update(seq_dets[fr_name], of_fw, of_bw, fr_name)
                seq_dets[fr_name] = new_dets

                assert(len(new_dets) == len(seq_dets[fr_name]))

            all_traj = mot_tracker.traj
            for t in all_traj:
                now_traj = all_traj[t]
                sc = 0
                for item in now_traj:
                    if item.idx == -1: continue

                    sc = max(sc, item.bbox.pos[4])
                    sc += item.bbox.pos[4]
                if len(now_traj) == 0: continue
                print(len(now_traj))
                #sc = sc * 1. / len(now_traj)
                display(now_traj, vid_name, data_path)
                for item in now_traj:
                    if item.idx == -1: continue
                    bbox = seq_dets[item.fr]
                    bbox[item.idx, 4] = sc
                    seq_dets[item.fr] = bbox


            for j in range(len(im_list)):
                fr_name = im_list[j].split('.')[0]
                new_dets = seq_dets[fr_name]
                for i in range(len(new_dets)):
                    d = new_dets[i]
                    print('%d,-1,%.6f,%.6f,%.6f,%.6f,%.6f,1,-1,-1'%(int(fr_name), float(d[0] + 1.), float(d[1] + 1.), 
                                                                float(d[2] - d[0]), float(d[3] - d[1]), float(d[4])),file=f)
