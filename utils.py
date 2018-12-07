import torch
from torch import svd

def gen_svd_vec(x, t=3):
    """ Generate SVD-vector
    :x - (N, C, H ,W) sized vector
    :t - Truncated value to cut off
    :svd_vector - would be (N, image_size*t*2 + t)
    """
    x = x.squeeze(1) # Remove channel
    batch_size, img_size, _ = x.size()
    svd_vector = torch.zeros((batch_size, t*(2*img_size+1)))
    for i, img in enumerate(x):
        u,s,v = svd(img)
        t_u,t_v,t_s = u[:,:t].reshape(-1), \
                      v.t()[:, :t].reshape(-1), \
                      s[:t]
        svd_vector[i] = torch.cat([t_u, t_s, t_v])
    return svd_vector




