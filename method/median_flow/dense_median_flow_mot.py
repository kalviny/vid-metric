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

def get_color(box):
    h = np.random.uniform(0.02, 0.31) + np.random.choice([0, 1./3, 2./3])
    l = np.random.uniform(0.3, 0.8)
    s = np.random.uniform(0.3, 0.8)

    rgb = colorsys.hls_to_rgb(h, l, s)
    return (int(rgb[0] * 256), int(rgb[1] * 256), int(rgb[2] * 256))


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

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w*h    #scale is just area
  r = w/h
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the form [x,y,s,r] and returns it in the form
    [x1,y1,x2,x2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class Bbox:
    def __init__(self):
        self.pos = []
        self.next_bbox = []
        self.prev_bbox = []
        self.sm_bbox = []
        self.frame = 0
        self.opt_flow = []
        self.ac_time = 1
        self.color = ()


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

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



class Sort(object):
    def __init__(self):
        self.trackers = []
        self.frame_count = 0
        self.dead_trk = []
        self.fail_cnt = 0

    def update(self, dets, of_fw, of_bw):
        '''
            dets - a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        '''

        new_dets = []

        self.frame_count += 1
        
        trks_of = []
        for b in self.trackers:
            trks_of.append(b.opt_flow)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks_of, 0.5)

        f_trk = []
        f_det = []
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0],0]
                # BW tracking
                bw_iou = iou(of_bw[d[0], :], trk.pos)
                if (bw_iou <= 0.5): 
                    self.fail_cnt += 1
                    f_trk.append(t)
                    f_det.append(d[0])
                    continue
                new_bbox = Bbox()
                new_bbox.pos = dets[d[0], :]
                new_bbox.opt_flow = of_fw[d[0], :]
                new_bbox.prev_bbox = trk.pos
                new_bbox.color = trk.color
                new_bbox.ac_time = trk.ac_time + 1

                #sm_bbox = KalmanBoxTracker(dets[d[0], 0:4])
                #sm_bbox.update(trk.opt_flow)
                sm_bbox = KalmanBoxTracker(trk.opt_flow)
                sm_bbox.update(dets[d[0], 0:4])
                new_bbox.sm_bbox = sm_bbox.get_state()[0]
                new_bbox.sm_bbox = np.hstack((new_bbox.sm_bbox, dets[d[0], 4]))

                x1 = (trk.opt_flow[0] + dets[d[0], 0]) / 2.
                y1 = (trk.opt_flow[1] + dets[d[0], 1]) / 2.
                x2 = (trk.opt_flow[2] + dets[d[0], 2]) / 2.
                y2 = (trk.opt_flow[3] + dets[d[0], 3]) / 2.

                new_dets.append(np.hstack((x1, y1, x2, y2, dets[d[0], 4])))

                self.trackers[t] = new_bbox

        # append dead trk_let in dead pool, and pop it out from active pool
        '''
        for t in range(len(self.trackers) - 1, -1, -1):
            if (t in unmatched_trks):
                self.dead_trk.append(self.trackers[t])
                del self.trackers[t]
        '''

        for t in range(len(self.trackers) - 1, -1, -1):
            if (t in unmatched_trks):
                if (self.trackers[t].pos[4] <= 0.9):
                    #self.dead_trk.append(self.trackers[t])
                    f_trk.append(t)
                    #del self.trackers[t]
                else:
                    trk = self.trackers[t]
                    new_bbox = Bbox()
                    new_bbox.pos = np.hstack((trk.opt_flow, trk.pos[4] * 0.8))
                    tmp = KalmanBoxTracker(trk.opt_flow)
                    new_opt_flow = tmp.predict()[0]
                    new_bbox.opt_flow = new_opt_flow
                    new_bbox.prev_box = trk.pos
                    new_bbox.ac_time = trk.ac_time + 1
                    new_bbox.color = trk.color
                    self.trackers[t] = new_bbox

        f_trk = list(set(f_trk))
        for t in range(len(self.trackers) - 1, -1, -1):
            if (t in f_trk):
                self.dead_trk.append(self.trackers[t])
                del self.trackers[t]

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
            new_bbox.color = get_color(dets[d, :])
            self.trackers.append(new_bbox)

        return np.array(new_dets)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # ------ acf result ----- #

    det_path = '/data01/kalviny/dataset/MOT/2015/detection/mot_acf_2015_proposal_300/mot_acf_2015_weighted_nms_pf/'
    bw_flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_acf_2015_proposal_300/mot_acf_2015_weighted_nms_pf/bw/'
    fw_flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_acf_2015_proposal_300/mot_acf_2015_weighted_nms_pf/fw/'
    out_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_acf_2015_proposal_300/mot_acf_2015_weighted_nms_mf_pf/'

    # --- frcnn result ----- #
    '''
    #det_path = '/data01/kalviny/dataset/MOT/2015/detection/mot_model0_2015_proposal_300/mot_model0_2015_weighted_nms/'
    det_path = '/data01/kalviny/dataset/MOT/2015/detection/mot_model0_2015_proposal_300/mot_model0_2015_weighted_nms_test/'

    bw_flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_model0_2015_proposal_300/mot_model0_2015_weighted_nms/bw/'
    fw_flow_path = '/data01/kalviny/dataset/MOT/2015/opt_flow/train/mot_model0_2015_proposal_300/mot_model0_2015_weighted_nms/fw/'

    out_path = '/home/kalviny/workspace/experiment/mx-faster-rcnn/result/mot_model0_2015_proposal_300/mot_model0_2015_weighted_nms_mf/'
    '''

    data_path = '/data01/kalviny/dataset/MOT/2015/train/'
    args = parse_args()
    display = args.display

    total_time = 0.0
    total_frames = 0

    colours = np.random.rand(32, 3) #used only for display

    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']

    mot_tracker = Sort()

    for vid_name in vid_list:
        print(vid_name)
        im_list = os.listdir(os.path.join(data_path, vid_name, 'img1'))
        im_list = sorted(im_list, key = lambda x: int(x.split('.')[0]))

        if not os.path.exists(os.path.join(out_path, vid_name)): os.makedirs(os.path.join(out_path, vid_name))
        out_dir = os.path.join(out_path, vid_name)
        with open(os.path.join(out_dir, 'det.txt'), 'w') as f:
            print(len(im_list))
            for j in range(len(im_list)):
                fr_name = im_list[j].split('.')[0]
                seq_dets = sio.loadmat(os.path.join(det_path, vid_name, '{}.mat'.format(fr_name)))
                dets = np.hstack((seq_dets['boxes'], seq_dets['zs']))

                try:
                    of_fw = sio.loadmat(os.path.join(fw_flow_path, vid_name, '{}.mat'.format(fr_name)))['boxes']
                    of_bw = sio.loadmat(os.path.join(bw_flow_path, vid_name, '{}.mat'.format(fr_name)))['boxes']
                except IOError:
                    of_fw = []
                    of_bw = []
                of_fw = np.array(of_fw)
                of_bw = np.array(of_bw)

                new_dets = mot_tracker.update(dets, of_fw, of_bw)

                #print(len(new_dets) - len(dets))

                assert(len(new_dets) == len(dets))
                #print(new_dets)

                for i in range(len(new_dets)):
                    d = new_dets[i]
                    print('%d,-1,%.6f,%.6f,%.6f,%.6f,%.6f,1,-1,-1'%(int(fr_name), float(d[0] + 1.), float(d[1] + 1.), 
                                                                    float(d[2] - d[0]), float(d[3] - d[1]), float(d[4])),file=f)

                '''
                for bbox in mot_tracker.trackers:
                        if (len(bbox.sm_bbox)):
                            d = bbox.sm_bbox
                        else:
                            d = bbox.pos
                        print('%d,-1,%.6f,%.6f,%.6f,%.6f,%.6f,1,-1,-1'%(int(fr_name), float(d[0] + 1.), float(d[1] + 1.), float(d[2] - d[0]), float(d[3] - d[1]), float(d[4])),file=f)

                '''

