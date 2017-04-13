import numpy as np

def IoU(a, b):
    """
        Compute as VOC IoU method
        a: x1, y1, x2, y2
        b: x1, y1, x2, y2
    """
    if (not len(a)) or (not len(b)): return 0

    x_min = np.maximum(a[0], b[0])
    y_min = np.maximum(a[1], b[1])
    x_max = np.minimum(a[2], b[2])
    y_max = np.minimum(a[3], b[3])
    w = np.maximum(0, x_max - x_min + 1.)
    h = np.maximum(0, y_max - y_min + 1.)
    inter = w * h
    uni = (a[2] - a[0] + 1.) * (a[3] - a[1] + 1.) + (b[2] - b[0] + 1.) * (b[3] - b[1] + 1.) - inter
    ov = inter / uni
    return ov

a = []
