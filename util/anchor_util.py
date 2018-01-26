import numpy as np

def whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w,h,x_ctr,y_ctr

def mkanchors(ws,hs,x_ctr,y_ctr):
    ws = ws[:,np.newaxis]
    hs = hs[:,np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                        y_ctr - 0.5 * (hs - 1),
                        x_ctr + 0.5 * (ws - 1),
                        y_ctr + 0.5 * (hs - 1)))
    return anchors

def ratio_enum(anchor,ratios):
    w,h,x_ctr,y_ctr = whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

def scale_enum(anchor,scales):
    w,h,x_ctr,y_ctr = whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

def generate_shift_matrix(height,width,stride,num_per_cell):
    h_shift = np.arange(0,width)
    h_shift = np.expand_dims(h_shift,axis = 0)
    h_shift = np.repeat(h_shift,height,axis = 0)
    h_shift = np.expand_dims(h_shift,axis = 2)
    h_shift = np.expand_dims(h_shift,axis = 3)
    h_shift = np.repeat(h_shift,num_per_cell,axis = 2)

    v_shift = np.arange(0,height)
    v_shift = np.expand_dims(v_shift,axis = 1)
    v_shift = np.repeat(v_shift,width,axis = 1)
    v_shift = np.expand_dims(v_shift,axis = 2)
    v_shift = np.expand_dims(v_shift,axis = 3)
    v_shift = np.repeat(v_shift,num_per_cell,axis = 2)

    zero = np.zeros(width)
    zero = np.expand_dims(zero,axis = 0)
    zero = np.repeat(zero,height,axis = 0)
    zero = np.expand_dims(zero,axis = 2)
    zero = np.expand_dims(zero,axis = 3)
    zero = np.repeat(zero,num_per_cell,axis = 2)

    right_shift = np.c_[h_shift,zero,h_shift,zero]
    down_shift = np.c_[zero,v_shift,zero,v_shift]

    shift = right_shift + down_shift
    shift = np.reshape(shift,(-1,num_per_cell,4))
    shift *= stride

    return shift

def ssd_scale(k,m):
    
    return 0.2 + 0.7/(m-1)*(k-1)
