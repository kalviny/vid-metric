import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

def process(y):
    for i in range(y.size - 1, 0, -1):
        y[i - 1] = np.maximum(y[i - 1], y[i])
    return y

if __name__ == '__main__':
    path = '../result/mot2015/accuracy/model0/'
    res_path = ['mot_model0_2015_nms_0.5', 'mot_model0_2015_weighted_nms']
    method = ['nms', 'weighted_nms']

    compare = 'nms_vs_weighted_nms'

    color = ['r', 'g', 'b', 'black']

    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']
    rcParams['font.family'] = 'Times New Roman'

    for i in range(len(vid_list)):
        vid_name = vid_list[i]
        print vid_name
        plt.figure(0)
        for j in range(len(method)):
            data = np.loadtxt(os.path.join(path + res_path[j], '{}.txt'.format(vid_name)), delimiter=',')
            data[0, 0] = 0.0
            data[0, 1] = 1.0
            plt.plot(data[:, 0], process(data[:, 1]), color[j], label = method[j])
        plt.xlabel('thr')
        plt.ylabel('acc')
        plt.grid(True)
        plt.legend(loc = 'lower left')
        plt.title('%s' % vid_name)
        out_dir = '../result/mot2015/figure/accuracy/' + compare
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, '{}.pdf'.format(vid_name)))
        plt.close(0)
