import os
import numpy as np
import scipy.io as sio
import json
import cPickle
from cython.bbox import bbox_overlaps_cython

def eval_rec(roidb, candidate_boxes=None):
    train_list = []
    rpn_of = '/data01/kalviny/dataset/MOT/2015/merge_set/merge_rpn_det/train/'

    with open('/data01/kalviny/dataset/MOT/2015/imglist/train_list', 'r') as f:
        train_list = f.readlines()

    area_range = [0**2, 1e5**2]
    gt_overlaps = np.zeros(0)
    num_pos = 0

    for i in range(len(roidb)):
        if not len(roidb[i]['boxes']): continue
        max_gt_overlaps = roidb[i]['gt_overlaps'].toarray().max(axis=1)
        gt_inds = np.where((roidb[i]['gt_classes'] > 0 & (max_gt_overlaps == 1)))[0]
        gt_boxes = roidb[i]['boxes'][gt_inds, :]
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
        valid_gt_inds = np.where((gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]
        num_pos += len(valid_gt_inds)

        if candidate_boxes is None:
            im_name = train_list[i].strip().split('_')
            v_name = im_name[0]
            fn_name = im_name[1]
            boxes = sio.loadmat(os.path.join(rpn_of, v_name, '{}.mat'.format(fn_name)))['boxes']
        else:
            boxes = candidate_boxes[i]
        if not boxes.shape[0]: continue

        overlaps = bbox_overlaps_cython(boxes.astype(np.float), gt_boxes.astype(np.float))
        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for j in range(gt_boxes.shape[0]):
            argmax_overlaps = overlaps.argmax(axis=0)
            max_overlaps = overlaps.max(axis=0)

            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert (gt_ovr >= 0)

            box_ind = argmax_overlaps[gt_ind]

            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert (_gt_overlaps[j] == gt_ovr)

            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
    gt_overlaps = np.sort(gt_overlaps)
    thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)
    recalls = np.zeros_like(thresholds)

    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    ar = recalls.mean()

    print 'average recall: {:.3f}'.format(ar)

    for threshold, recall in zip(thresholds, recalls):
        print 'recall @{:.2f}: {:3f}'.format(threshold, recall)


if __name__ == '__main__':
    roidb_path = '/home/kalviny/workspace/video_detection/video-detection/detection/faster_rcnn_vid/mx-faster-rcnn/data/cache/'
    rpn_bbox = '/home/kalviny/workspace/video_detection/video-detection/detection/faster_rcnn_vid/mx-faster-rcnn/data/rpn_data/'
    bbox = []
    with open(os.path.join(rpn_bbox, 'mot_train_rpn.pkl.bak'), 'rb') as f:
        bbox = cPickle.load(f)
    roidb = []
    with open(os.path.join(roidb_path, 'mot_train_gt_roidb.pkl'), 'rb') as f:
        roidb = cPickle.load(f)

    eval_rec(roidb, bbox)
