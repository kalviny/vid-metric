import os
import numpy as np
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':

    path = 'result/mot2015/traj/'
    res_path = ['mot_model0_2015_nms', 'mot_model0_2015_weighted_nms', 'mot_model0_2015_with_tr_nms', 'mot_model0_2015_with_tr_weighted_nms']
    method = ['nms', 'weighted_nms', 'nms + MF', 'weighted_nms + MF']
    color = ['r', 'g', 'b', 'black']
    vid_list = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof',
                'ETH-Sunnyday', 'KITTI-13', 'KITTI-17']

    y_label = ['Fragment', 'Center', 'Ratio', 'Stability']

    for i in range(len(vid_list)):
        vid_name = vid_list[i]
        print 'processing %s' % (vid_name)

        plt.figure(1, figsize=(10, 10))
        #plt.title(vid_name)

        for j in range(len(method)):
            data = np.loadtxt(os.path.join(path + res_path[j], '{}.txt'.format(vid_name)), delimiter=',')
            #thr, F_err, C_err, R_err

            for x in range(len(data[:, 1])):
                if math.isnan(data[x, 1]): data[x, 1] = 0.
                if math.isnan(data[x, 2]): data[x, 2] = 0.
                if math.isnan(data[x, 3]): data[x, 3] = 0.
            
            tmp = np.array(data[:, 1] + data[:, 2] + data[:, 3])
            data = np.column_stack((data, tmp))

            for r in range(1, 5):
                plt.subplot(2, 2, r)
                plt.plot(data[:, 0], data[:, r], color[j], label = method[j])
                plt.xlabel('Threshold')
                plt.ylabel(y_label[r - 1])

        plt.suptitle(vid_name, fontsize=20)
        out_dir = 'result/mot2015/output/model0/'
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, '{}.png'.format(vid_name)))
        plt.close(1)
