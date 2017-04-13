import os
import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    res_dir = '../result/mot2016/output/'

    vid_list = os.listdir(res_dir)
    vid_list = sorted(vid_list)

    for i in range(len(vid_list)):
        vid_name = vid_list[i].split('.')[0]
        print 'processing %s' % (vid_name)
        data = np.loadtxt(os.path.join(res_dir, '{}.txt'.format(vid_name)), delimiter=',')

        thr = data[:, 0]
        F_err = data[:, 1]
        C_err = data[:, 2]
        R_err = data[:, 3]

        for j in range(len(F_err)):
            if math.isnan(F_err[j]): F_err[j] = 0.
            if math.isnan(C_err[j]): C_err[j] = 0.
            if math.isnan(R_err[j]): R_err[j] = 0.

        '''
        fig = plt.figure(0)
        ax1 = fig.add_subplot(111)
        ax1.plot(thr, C_err, 'g--', label = 'Cneter_err')
        ax1.plot(thr, R_err, 'b--', label = 'Ratio_err')
        ax1.set_ylabel('Stability')
        ax1.set_ylim([0, 1.])
        ax1.set_title('%s' % vid_name)

        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(thr, F_err, 'r--', label = 'Fragment')
        ax2.set_xlim([0, 1.])
        ax2.set_ylabel('Number')
        ax2.set_xlabel('Threshold')

        #plt.show()

        plt.savefig(os.path.join('../result/mot2016/traj/' '{}.png'.format(vid_name)))

        plt.close(0)
        '''

        plt.figure(0)
        plt.plot(thr, F_err, 'r--', label='F_err')
        plt.plot(thr, C_err, 'g--', label = 'Center_err')
        plt.plot(thr, R_err, 'b--', label = 'Ratio_err')
        plt.legend(loc='upper left')
        plt.xlabel('threshold')
        plt.ylabel('error')
        plt.title('%s' % vid_name)
        plt.savefig(os.path.join('../result/mot2016/traj/' '{}.png'.format(vid_name)))
        plt.close(0)
