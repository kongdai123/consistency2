import os,os.path
os.environ['PATH'] = "/usr/local/cuda/bin:" + os.environ['PATH']
import numpy as np
import torch
import nvdiffrast.torch as dr
import json
from consistency2.ViewDataset import ViewDataset
from consistency2.mesh import RenderSettings, Mesh
from consistency2.cubemap import CubeMap
from consistency2.vis import *
from consistency2.generate import init_pipeline

import argparse

if '__main__' == __name__:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesh_config",
        type=str,
        default="./config/mesh_config.json",
        help="path to mesh config",
    )

    parser.add_argument(
        "--view_config",
        type=str,
        default="./config/view_config.json",
        help="path to view config",
    )

    parser.add_argument(
        "--paint_config",
        type=str,
        default="./config/paint_config.json",
        help="path to painting config",
    )

    parser.add_argument('--no_video', action='store_false', dest='video',
                        help='do not output 360 video')

    # load configs

    args = parser.parse_args()

    paint_config = args.paint_config
    with open(paint_config) as f:
        paint_config = json.load(f)

    mesh_config = args.mesh_config
    with open(mesh_config) as f:
        mesh_config = json.load(f)


    view_config = args.view_config
    with open(view_config) as f:
        view_config = json.load(f)


    # load renderer and pipeline

    device = "cuda"
    glctx = dr.RasterizeCudaContext()

    dtype = torch.float16

    pipe = init_pipeline()

    # env map

    env_map_path = paint_config["env_map"]
    env =np.load(env_map_path)
    env = torch.as_tensor(env, device='cuda')[None]

    cm = CubeMap(env, H = 1024, W = 1024)

    # mesh  

    meshes = Mesh(mesh_config)

    # view dataset
    views = view_config["views"]
    elev_angs = view_config["elev"]
    azim_angs = view_config["azim"]

    fov = view_config["fov"]
    dist = view_config["dist"]

    prompt = mesh_config["prompt"]
    prompts = f"a high resolution photo of a {prompt}, ultra quality, extra sharp, crisp, Fujifilm XT3, 8K UHD"

    vd = ViewDataset(elev_angs, azim_angs, dist, prompts, fov = fov)

    # view dataset

    save_dir =  mesh_config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # generate settings

    rendersettings_latent = RenderSettings(resolution = 128, interp_type="linear-gp", latent = True)
    rendersettings_cond = RenderSettings(resolution = 1024, interp_type="linear-gp", latent = True)

    guidance_scale = paint_config.get("guidance_scale", 7.5)
    num_inference_steps = paint_config.get("num_inference_steps", 4)
    noise_res = paint_config.get("noise_res", 768)
    color_res = paint_config.get("color_res", 2048)
    seed_latents = mesh_config.get("seed_latents", 2024) 

    controlnet_conditioning_scales = paint_config.get("condition_scales", 0.4)

    print(f"guidance_scale {guidance_scale}")
    print(f"num_inference_steps {num_inference_steps}")
    print(f"noise_res {noise_res}")
    print(f"color_res {color_res}")
    print(f"condition_scales {controlnet_conditioning_scales}")

    # generate

    latents = pipe(
        meshes,
        cm,
        vd,
        glctx,
        rendersettings_latent,
        rendersettings_cond, 
        num_inference_steps=num_inference_steps, 
        controlnet_conditioning_scales=controlnet_conditioning_scales, 
        save_dir = save_dir,
        guidance_scale = guidance_scale,
        rand_add_noise= False,
        rotate_camera= True,
        seed_latents = seed_latents,
        noise_res = noise_res,
        color_res = color_res
    )

    img_all = save_combined(save_dir, latents, vd, pipe)

    rendersettings_mesh = RenderSettings(interp_type="mip", latent = False, resolution = 1024)

    tex_opt = meshes.optimize_texture(glctx, img_all, rendersettings_mesh, view_dataset = vd, tex_opt_shape = (2048,2048,3), max_iter = 2000,  lr_base = 1e-3, clamp = True)
    tex_opt = tex_opt.detach() 
    meshes.save_obj(f"{save_dir}/out.obj", tex = tex_opt)

    #save a 360 video

    if args.video:
        print("saving 360 view video")
        views_vis = 360
        elev_angs_vis = [0] * views_vis
        azim_angs_vis = [i * (360/views_vis)  for i in range(views_vis)]

        dist_vis = [view_config["dist"]] * views_vis
        vd_vis = ViewDataset(elev_angs_vis, azim_angs_vis, dist_vis, prompts, fov = fov)

        output_video(f"{save_dir}/output_col_opt", meshes, vd_vis, glctx,rendersettings_mesh, tex = tex_opt, cm = cm)

