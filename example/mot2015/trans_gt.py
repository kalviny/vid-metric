import os
import cPickle
import glob
import json
from itertools import groupby

def conv_bbox(bbox):
    res_box = []
    res_cls = []
    res_traj = []
    for item in bbox:
        b = item.strip().split(',')
        new_box = []
        if (float(b[6]) == 0): continue
        new_box.append(float(b[2]) - 1.)
        new_box.append(float(b[3]) - 1.)
        new_box.append(float(b[2]) + float(b[4]) - 1.)
        new_box.append(float(b[3]) + float(b[5]) - 1.)
        #new_box.append(int(b[6]))
        #new_box.append(b[7]) # use only for 2016
        res_box.append(new_box)
        res_cls.append('1')
        res_traj.append(b[1])

    return res_box, res_cls, res_traj


def load_annotations(imageset, mot_path):

    for v_name in sorted(glob.glob(os.path.join(mot_path, imageset, '*'))):
        im_list = []
        im_name = os.listdir(os.path.join(mot_path, v_name, 'img1'))
        im_name = [str(int(x.split('.')[0])) for x in im_name]
        print v_name, len(im_name)
        if (imageset == 'train'):
            with open(os.path.join(v_name, 'gt', 'gt.txt'), 'r') as f:
                line = f.readlines()
                line = [x.strip() for x in line]
                line = sorted(line, key=lambda x: x.split(',')[0])
                bbox = []
                fn_idx = []
                for k, g in groupby(line, lambda x: x.split(',')[0]):
                    bbox.append(list(g))
                    fn_idx.append(k)

                im_name = sorted(im_name, key = lambda x: int(x))
                fn_idx = sorted(fn_idx, key = lambda x: int(x))

                for i in range(len(im_name)):
                    sub_im = {}
                    sub_im['img'] = im_name[i]
                    sub_im['bbox'] = []
                    sub_im['cls'] = []
                    sub_im['traj_id'] = []
                    if (im_name[i] in fn_idx):
                        x = fn_idx.index(im_name[i])
                        sub_im['bbox'], sub_im['cls'], sub_im['traj_id'] = conv_bbox(bbox[x])
                    im_list.append(sub_im)
                out_file = os.path.join(v_name, 'gt', 'gt.json')
                js_file = json.dumps(im_list)
                with open(out_file, 'w') as fout:
                    fout.write(js_file)
if __name__ == '__main__':
    load_annotations('train', '/data01/kalviny/dataset/MOT/2015')
    #load_annotations('test', '/data01/kalviny/dataset/MOT/2015')
