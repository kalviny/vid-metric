import os
import numpy as np
import scipy.io as sio

def py_weighted_nms(dets, thresh_lo, thresh_hi):
    """
    voting boxes with confidence > thresh_hi
    keep boxes overlap <= thresh_lo
    rule out overlap > thresh_hi
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh_lo: retain overlap <= thresh_lo
    :param thresh_hi: vote overlap > thresh_hi
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order] - inter)

        inds = np.where(ovr <= thresh_lo)[0]
        inds_keep = np.where(ovr > thresh_hi)[0]
        if len(inds_keep) == 0:
            break

        order_keep = order[inds_keep]
        x1_avg = np.sum(scores[order_keep] * x1[order_keep]) / np.sum(scores[order_keep])
        y1_avg = np.sum(scores[order_keep] * y1[order_keep]) / np.sum(scores[order_keep])
        x2_avg = np.sum(scores[order_keep] * x2[order_keep]) / np.sum(scores[order_keep])
        y2_avg = np.sum(scores[order_keep] * y2[order_keep]) / np.sum(scores[order_keep])

        keep.append([x1_avg, y1_avg, x2_avg, y2_avg, scores[i]])
        order = order[inds]
    return np.array(keep)

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

if __name__ == '__main__':
    vid = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
            'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']
    root_path = '/data01/kalviny/dataset/MOT/2015/detection/proposal_flow/mot_model0_2015_proposal_300/'

    method = 'mot_model0_2015_without_nms/window_size_3_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow_decay_0.9/'

    proposal_path = root_path + method

    for vid_name in vid:
        vid_path = os.path.join(proposal_path, vid_name)
        im_list = os.listdir(os.path.join(vid_path))
        im_list = sorted(im_list, key = lambda x: int(x.split('.')[0]))
        vid_out = os.path.join('result/', vid_name)
        if not os.path.exists(vid_out):
            os.mkdir(vid_out)
        print 'processing %s' % (vid_name)
        print vid_out, len(im_list)
        #with open(os.path.join(vid_out, 'det.txt'), 'w') as f:
        f = open(str(vid_out) + '/det.txt', 'w')
        for im_name in im_list:
            #im_name = im_name.split('.')[0]
            data = sio.loadmat(os.path.join(proposal_path, vid_name, im_name))
            im_data = data['boxes']
            im_score = data['zs']
            bbox = np.hstack((im_data, im_score))

            #keep = py_cpu_nms(bbox, 0.5)
            #bbox = bbox[keep, :]

            keep = py_weighted_nms(bbox, 0.5, 0.7)
            bbox = keep
            bbox[:, 2:4] -= bbox[:, 0:2]
            bbox[:, 0:2] += 1
            for i in range(len(bbox)):
                _name = int(im_name.split('.')[0])
                f.write('%d,-1,%.6f,%.6f,%.6f,%.6f,%.6f,-1,-1,-1\n' % (_name, bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3], bbox[i, 4]))
