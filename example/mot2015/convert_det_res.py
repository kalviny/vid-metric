from __future__ import print_function
import os

if __name__ == '__main__':
    det_path = '/home/kalviny/workspace/video_detection/video-detection/detection/faster_rcnn_vid/mx-faster-rcnn/data/MOT/2015/results/'
    vid_list = os.listdir(det_path)

    for vid_name in vid_list:
        with open(os.path.join(det_path, vid_name, '{}.txt'.format(vid_name)), 'r') as f:
            fout = open(os.path.join(det_path, vid_name, 'det.txt'), 'w')
            data = f.readlines()
            data = [x.strip() for x in data]
            for x in range(len(data)):
                y = data[x].split(',')
                y[0] = y[0].split('.')[0]
                print(','.join(y), file=fout)
            fout.close()
