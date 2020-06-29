# -*- coding: utf-8 -*-
import numpy as np


def voxel_downsampling(raw_vxl, factor=0.5):
    h, w, d = raw_vxl.shape

    dh, dw, dd = int(h * factor), int(w * factor), int(d * factor)
    dst_vxl = np.zeros((dh, dw, dd), dtype=np.int8)
    s = int(1 / factor)

    for i in range(dh):
        for j in range(dw):
            for k in range(dd):
                key_sum = np.sum(raw_vxl[i * s:(i + 1) * s, j * s:(j + 1) * s, k * s:(k + 1) * s])
                dst_vxl[i, j, k] = 1 if key_sum >= 1.0 else 0

    return dst_vxl