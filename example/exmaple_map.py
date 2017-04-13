import os
import numpy as np
import json
import cPickle
from eval_map import do_eval

cache_dir = "/data01/kalviny/dataset/VOC/"

def parse_voc_rec(filename):
    import xml.etree.ElementTree as ET
    tree = ET.parse(os.path.join('{}.xml').format(filename))
    objects = []
    for obj in tree.findall('object'):
        obj_dict = {}
        obj_dict['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_dict)
    return objects

if __name__ == '__main__':
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, 'annotations.pkl')
    anno_path = '/home/seanlx/Dataset/VOCdevkit/VOC2007/Annotations/'
    imageset_file = '/home/seanlx/Dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_files = [x.strip() for x in lines]
    if not os.path.isfile(cache_file):
        recs = {}
        for ind, im_name in enumerate(image_files):
            im_name = im_name.split('.')[0]
            recs[im_name] = parse_voc_rec(os.path.join(anno_path, im_name))
            #print recs[im_name]
        with open(cache_file, 'w') as f:
            cPickle.dump(recs, f)
    else:
        with open(cache_file, 'r') as f:
            recs = cPickle.load(f)
    cls_name = 'bird'
    #print recs
    anno = {}
    for im_name in image_files:
        im_name = im_name.split('.')[0]
        objects = [obj for obj in recs[im_name] if obj['name'] == cls_name]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)
        anno[im_name] = {'bbox': bbox,
                         'difficult': difficult,
                         'det': det}
    with open('bird.txt', 'r') as f:
        lines = f.readlines()
    det  = [x.strip().split(' ') for x in lines]
    det = np.array(det)
    #det[:, [1, 5]] = det[:, [5, 1]]

    res = do_eval(anno, det)
    print res[-1]
    


