"""
    Brief: Compute similarity metrics for evaluation
    Author: wangpeng54@baidu.com
    Date: 2018/6/20
"""

import numpy as np
import utils.utils as uts
import car_models
import cv2
from tqdm import tqdm
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, look_at_view_transform,
    MeshRasterizer, BlendParams, SoftSilhouetteShader, TexturesVertex
)
import torch
from pytorch3d.structures import Meshes

# def render_car_light(pose, vertices, face):
#     """Render a car instance given pose and car_name
#     """
#     scale = np.ones((3, ))
#     pose = np.array(pose)
#     vert = uts.project(pose, scale, vertices)
#     intrinsic = np.float64([128, 128, 64, 64])
#     depth, mask = render.renderMesh_py(np.float64(vert),
#                                         np.float64(face),
#                                         intrinsic,
#                                         128,
#                                         128,
#                                         np.float64(0))

#     return mask


def compute_reproj_light(car_vertices, car_face, gt_car_ids, dt_car_ids, dt_vertices):
    """Compute reprojection error between two cars
    """
    sims_mat = np.zeros([len(dt_vertices),len(gt_car_ids)])

    # redering setting
    cameras = FoVPerspectiveCameras(device = 'cuda')
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
    )

    silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    verts_rgb = torch.ones((1,)+dt_vertices[0].shape).to('cuda')
    textures = TexturesVertex(verts_features=verts_rgb)

    for dt_i, dt_vert in tqdm(enumerate(dt_vertices), total = len(dt_vertices)):
        vert = torch.from_numpy(dt_vert).to('cuda')
        vert[:, [0, 1]] *= -1

        dt_masks = np.zeros((10,128,128))
        for i, rot in enumerate(np.linspace(-np.pi, np.pi, num=10)):
            R, T = look_at_view_transform(3, azim=rot, device='cuda', degrees=False)
            
            mesh = Meshes(verts=vert.unsqueeze(0), faces=car_face.unsqueeze(0), textures=textures)
            mask = silhouette_renderer(meshes_world=mesh, R=R, T=T).cpu().numpy()[0][...,3]
            dt_masks[i] = mask
        
        for gt_i, gt_car_id in enumerate(gt_car_ids):
            gt_masks = car_vertices[gt_car_id]
            sims = np.zeros(10)
            
            for i in range(10):
                mask1 = dt_masks[i]
                mask2 = gt_masks[i]
                sims[i] = IOU(mask1, mask2)
            sims_mat[dt_i][gt_i] = np.mean(sims)

    return sims_mat

def compute_reproj(car_vertices, car_face, gt_car_ids, dt_car_ids, dt_vertices):
    """Compute reprojection error between two cars
    """
    sims_mat = np.zeros([len(dt_vertices),len(gt_car_ids)])

    # redering setting
    cameras = FoVPerspectiveCameras(device = 'cuda')
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
    image_size=1280, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
    )

    silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    verts_rgb = torch.ones((1,)+dt_vertices[0].shape).to('cuda')
    textures = TexturesVertex(verts_features=verts_rgb)
           
    for dt_i, dt_vert in tqdm(enumerate(dt_vertices), total = len(dt_vertices)):
        vert = torch.from_numpy(dt_vert).to('cuda')
        vert[:, [0, 1]] *= -1

        dt_masks = np.zeros((100,1280,1280))
        for i, rot in enumerate(np.linspace(-np.pi, np.pi, num=100)):
            R, T = look_at_view_transform(3, azim=rot, device='cuda', degrees=False)
            mesh = Meshes(verts=vert.unsqueeze(0), faces=car_face.unsqueeze(0), textures=textures)
            mask = silhouette_renderer(meshes_world=mesh, R=R, T=T).cpu().numpy()[0][...,3]

            dt_masks[i] = mask
        
        for gt_i, gt_car_id in enumerate(gt_car_ids):
            gt_masks = car_vertices[gt_car_id]
            sims = np.zeros(100)
            
            for i in range(100):
                mask1 = dt_masks[i]
                mask2 = gt_masks[i]
                sims[i] = IOU(mask1, mask2)
                
            sims_mat[dt_i][gt_i] = np.mean(sims)
    return sims_mat

def pose_similarity(dt, gt, dt_vertices = None, car_vertices = None, car_face = None, light = True):
    """compute pose similarity in terms of shape, translation and rotation
    Input:
        dt: np.ndarray detection [N x 7] first 6 dims are roll, pitch, yaw, x, y, z
        gt: save with dt

    Output:
        sim_shape: similarity based on shape eval
        dis_trans: distance based on translation eval
        dis_rot:   dis.. based on rotation eval
    """
    dt_num = len(dt)
    gt_num = len(gt)

    dt_car_id = np.uint32(dt[:, -1])
    gt_car_id = np.uint32(gt[:, -1])
    car_face = torch.from_numpy(car_face).to('cuda')

    if light:
        sims_shape = compute_reproj_light(car_vertices, car_face, gt_car_id, dt_car_id,dt_vertices)
    else:
        sims_shape = compute_reproj(car_vertices, car_face, gt_car_id, dt_car_id,dt_vertices)
    # translation similarity
    dt_car_trans = dt[:, 3:-1]
    gt_car_trans = gt[:, 3:-1]

    tilde_dt = np.tile(dt_car_trans[:, None, :], [1, gt_num, 1])
    tilde_gt = np.tile(gt_car_trans[None, :, :], [dt_num, 1, 1])  

    #relative
    rel_dis_trans = np.linalg.norm(tilde_dt - tilde_gt, axis=2,ord=1)/np.linalg.norm(tilde_gt, axis=2,ord=2)
    dis_trans = np.linalg.norm(tilde_dt - tilde_gt, axis=2)

    # rotation similarity
    dt_car_rot = uts.euler_angles_to_quaternions(dt[:, :3])
    gt_car_rot = uts.euler_angles_to_quaternions(gt[:, :3])
    q1 = dt_car_rot / np.linalg.norm(dt_car_rot, axis=1)[:, None]
    q2 = gt_car_rot / np.linalg.norm(gt_car_rot, axis=1)[:, None]


    # diff = abs(np.matmul(q1, np.transpose(q2)))
    diff = abs(1 - np.sum(np.square(np.tile(q1[:, None, :], [1, gt_num, 1]) - \
            np.tile(q2[None, :, :], [dt_num, 1, 1])), axis=2) / 2.0)
    dis_rot = 2 * np.arccos(diff) * 180 / np.pi
    
    return sims_shape, dis_rot, dis_trans, rel_dis_trans


def IOU(mask1, mask2):
    """ compute the intersection of union of two logical inputs
    Input:
        mask1: the first mask
        mask2: the second mask
    """

    inter = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    if np.sum(inter) == 0:
        return 0.

    return np.float32(np.sum(inter)) / np.float32(np.sum(union))





if __name__ == '__main__':
    shape_sim_mat = np.loadtxt('./test_eval_data/sim_mat.txt')
    fake_gt = []
    fake_dt = []
    sim_shape, sim_trans, sim_rot = pose_similarity(fake_dt, fake_gt, shape_sim_mat)


