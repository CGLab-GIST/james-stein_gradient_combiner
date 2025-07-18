import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import drjit as dr
from util import render_ref, get_relMSE
import numpy as np
import os
import torch

"""
MITSUBA3 version == '3.5.2'
"""


import random
seed = 2021
deterministic = True

random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="frame", help='frame, tray, plane')
parser.add_argument("--iter", type=int, default=400)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--spp", type=int, default=16)
parser.add_argument("--backward_spp", type=int, default=16)
parser.add_argument("--winSize", type=int, default=21)
parser.add_argument("--loss", type=str, choices=['L1', 'L2', 'RelativeL2'], default="L2")
parser.add_argument("--use_prb", action="store_true")
parser.add_argument("--use_biased", action="store_true")
parser.add_argument("--use_cb", action="store_true")
parser.add_argument("--use_gaussian", action="store_true")
parser.add_argument("--use_js", action="store_true")

args, unknown = parser.parse_known_args()


print("[Learning Rate: %.4f]"%args.lr)

if args.scene == "frame":
    scene_dir = './scenes/frame'
    scene_name = 'scene.xml'
    key = 'frame_picture_bsdf.brdf_0.reflectance.data'
    resX = 1024
    resY = 1024
    TEST_SPP = 1024
    MAX_DEPTH = 13
    INIT_PARAM = 0.5
    BETA_1 = 0.2
    BETA_2 = 1 - (1 - BETA_1)**2
elif args.scene == "tray":
    scene_dir = './scenes/tray'
    scene_name = 'scene.xml'
    key = 'bsdf_tray.nested_bsdf.metallic.data'
    resX = 1024
    resY = 1024
    TEST_SPP = 1024
    MAX_DEPTH = 13
    INIT_PARAM = 0.2
    BETA_1 = 0.2
    BETA_2 = 1 - (1 - BETA_1)**2
elif args.scene == "plane":
    scene_dir = './scenes/plane'
    scene_name = 'scene.xml'
    key = 'plane.bsdf.alpha.data'
    resX = 1280
    resY = 1280
    TEST_SPP = 8192
    MAX_DEPTH = 5
    INIT_PARAM = 0.2
    BETA_1 = 0.2
    BETA_2 = 1 - (1 - BETA_1)**2


scene_path = os.path.join(scene_dir, scene_name)
scene = mi.load_file(scene_path, resx=resX, resy = resY)

path_integrator = mi.load_dict({'type': 'path', 'max_depth': MAX_DEPTH})
prb_integrator = mi.load_dict({'type': 'prb', 'max_depth': MAX_DEPTH})

params = mi.traverse(scene)
param_ref = mi.TensorXf(params[key])
param_shape = np.array(params[key].shape)


### render a target image
gt_path = os.path.join(scene_dir, 'target_img.exr')
mi.util.write_bitmap("test.exr", param_ref)
if not os.path.exists(gt_path):
    print("[RENDER GT]")
    target_image = render_ref(scene, TEST_SPP, prb_integrator, resY, resX)
    mi.util.write_bitmap(os.path.join(scene_dir, 'target_img.exr'), target_image)
else:
    target_image = mi.Bitmap(gt_path).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
    target_image = mi.TensorXf(target_image)   


def run_prb_optimization():
    from prb_util.prb_optimizers import CustomAdam

    dr.set_flag(dr.JitFlag.KernelHistory, True)

    opt = CustomAdam(lr=args.lr, beta_1=BETA_1, beta_2 = BETA_2)
    np.random.seed(0)
    print("[beta_1: %.3f, beta_2: %.3f]"%(BETA_1, BETA_2))

    """
        set initial parameters
    """
    param_initial = np.full(param_shape, INIT_PARAM).astype(np.float32)

    params[key] = mi.TensorXf(param_initial)
    params.update();

    """
        update trainable parameters to the optimizer
    """
    opt[key] = params[key]
    params.update(opt)

    ### render a initial image
    init_path = os.path.join(scene_dir, 'init_img.exr')
    if not os.path.exists(init_path):
        print("[RENDER INITIAL IMAGE]")
        intial_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
        mi.util.write_bitmap(os.path.join(scene_dir, 'init_img.exr'), intial_image)
    
    for it in range(args.iter):

        seed_0 = np.random.randint(2**31)
        seed_1 = np.random.randint(2**31)

        image =  mi.render(scene, params, integrator=prb_integrator, spp=args.spp, spp_grad = args.spp, seed = seed_0, seed_grad = seed_1)

        if args.loss == "L2":
            loss = dr.mean(dr.sqr(image - target_image))
        elif args.loss == "RelativeL2":
            loss = dr.mean(dr.sqr(image - target_image) / (dr.sqr(target_image) + 1e-2))
        elif args.loss == "L1":
            loss = dr.mean(dr.abs(image - target_image))

        dr.backward(loss)
        opt.step()
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)
        params.update(opt)

        print("[Itertaion %d] Loss : %.4f"%(it, loss[0]))


    with dr.suspend_grad():

        final_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
        final_image = np.where(np.isnan(final_image), 0., final_image)
        mi.util.write_bitmap(os.path.join(scene_dir, "prb_target_img_iter%d_%dspp_%s_beta%d_lr%d.exr")%(args.iter, args.spp, args.loss, BETA_1*10, args.lr * 100), final_image)
        error = get_relMSE(final_image, target_image)
        print("[PRB - RelMSE] %.5f"%error)



def run_combine_optimization():
    from prb_util.prb_optimizers import CustomAdamJS

    opt = CustomAdamJS(lr=args.lr, beta_1=BETA_1, beta_2 = BETA_2,
           use_js=args.use_js, use_gaussian=args.use_gaussian, use_cb=args.use_cb)
    print("[beta_1: %.3f, beta_2: %.3f]"%(BETA_1, BETA_2))

    np.random.seed(0)

    """
        set initial parameters
    """

    param_initial = np.full(param_shape, INIT_PARAM).astype(np.float32)
    params[key] = mi.TensorXf(param_initial)
    params.update()

    """
        update trainable parameters to the optimizer
    """
    opt[key] = params[key]
    params.update(opt);

    ### render a initial image
    init_path = os.path.join(scene_dir, 'render_init.exr')
    if not os.path.exists(init_path):
        print("[RENDER INITIAL IMAGE]")
        intial_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
        mi.util.write_bitmap(os.path.join(scene_dir, 'render_init.exr'), intial_image)
    

    for it in range(args.iter):
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
      
            image = mi.render(scene, params, integrator=prb_integrator, spp=args.spp, spp_grad = args.spp, seed = it, seed_grad=(it + args.iter))
            
            if args.loss == "L2":
                loss = dr.mean(dr.sqr(image - target_image))
            elif args.loss == "RelativeL2":
                loss = dr.mean(dr.sqr(image - target_image) / (dr.sqr(target_image) + 1e-2))
            elif args.loss == "L1":
                loss = dr.mean(dr.abs(image - target_image))

            dr.backward(loss)


            opt.step()
            opt[key] = dr.clamp(opt[key], 0.0, 1.0)
            params.update(opt)

            print("[Itertaion %d] Loss : %.4f"%(it, loss[0]))


    with dr.suspend_grad():
        if args.use_js:
            if args.use_cb:
                final_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
                final_image = np.where(np.isnan(final_image), 0., final_image)      

                mi.util.write_bitmap(os.path.join(scene_dir, "prb_cb_pre_js_img_iter%d_%dspp_%s_beta%d.exr")%(args.iter, args.spp, args.loss, BETA_1*10), final_image)
                error = get_relMSE(final_image, target_image)
                print("[CB_JS - RelMSE] %.5f"%error)

            elif args.use_gaussian:
                final_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
                final_image = np.where(np.isnan(final_image), 0., final_image)                 
                mi.util.write_bitmap(os.path.join(scene_dir, "prb_gaussian_js_img_iter%d_%dspp_%s_beta%d.exr")%(args.iter, args.spp, args.loss, BETA_1*10), final_image)
                error = get_relMSE(final_image, target_image)
                print("[Gaussain_JS - RelMSE] %.5f"%error)

        else:
            if args.use_cb:
                # mi.util.write_bitmap(os.path.join(scene_dir, "prb_cb_pre_param_iter%d_%dspp_%s_beta%d.exr")%(args.iter, args.spp, args.loss, BETA_1*10), params[key])
                final_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
                final_image = np.where(np.isnan(final_image), 0., final_image)      
                mi.util.write_bitmap(os.path.join(scene_dir, "prb_cb_pre_img_iter%d_%dspp_%s_beta%d.exr")%(args.iter, args.spp, args.loss, BETA_1*10), final_image)
                error = get_relMSE(final_image, target_image)
                print("[CB - RelMSE] %.5f"%error)

            elif args.use_gaussian:
                final_image = render_ref(scene, TEST_SPP, path_integrator, resY, resX)
                final_image = np.where(np.isnan(final_image), 0., final_image)      
                mi.util.write_bitmap(os.path.join(scene_dir, "prb_gaussian_img_iter%d_%dspp_%s_beta%d.exr")%(args.iter, args.spp, args.loss, BETA_1*10), final_image)
                error = get_relMSE(final_image, target_image)
                print("[Gaussain - RelMSE] %.5f"%error)

            
if args.use_prb:
    run_prb_optimization()
elif args.use_biased:
    run_combine_optimization()

