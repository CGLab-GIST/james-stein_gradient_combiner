import torch
import numpy as np
import mitsuba as mi
import drjit as dr
import time

def get_MSE(input, ref):
    l2 = np.square(np.subtract(input, ref))
    return np.mean(l2)

def get_relMSE(input, ref):
    eps = 1e-2
    num = np.square(np.subtract(input, ref))
    denom = np.mean(ref, axis=-1, keepdims=True)
    relMse = num / (denom * denom + eps)
    relMseMean = np.mean(relMse)
    return relMseMean



def render_ref(scene, render_spp, integrator, resy, resx):
    with dr.suspend_grad():
        num_render = int(render_spp / 512)
        image = dr.zeros(mi.TensorXf, (resy, resx, 3))
        for i in range(0, num_render):
            # print(i)
            image_sub = mi.render(scene, spp=512, seed = i, integrator=integrator)
            image += image_sub
        image /= num_render
    return image

