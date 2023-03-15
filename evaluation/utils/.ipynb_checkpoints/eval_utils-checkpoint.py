"""
    Brief: Compute similarity metrics for evaluation
    Author: wangpeng54@baidu.com
    Date: 2018/6/20
"""

import numpy as np
import utils.utils as uts
import car_models
import renderer.render_egl as render
import cv2
from tqdm import tqdm


def render_car(pose, vertices, face):
    """Render a car instance given pose and car_name
    """
    scale = np.ones((3, ))
    pose = np.array(pose)
    vert = uts.project(pose, scale, vertices)
    intrinsic = np.float64([395, 395, 640, 640])
    depth, mask = render.renderMesh_py(np.float64(vert),
                                        np.float64(face),
                                        intrinsic,
                                        1280,
                                        1280,
                                        np.float64(0))

    return mask

def compute_reproj(car_vertices, car_face, gt_car_ids, dt_vertices):
    """Compute reprojection error between two cars
    """
    sims_mat = np.zeros([len(dt_vertices),len(gt_car_ids)])

    for dt_i, dt_vert in tqdm(enumerate(dt_vertices), total = len(dt_vertices)):
        
        vert = np.array(dt_vert)
        # vert[:, [0, 1]] *= -1
        center = np.mean(vert, axis = 0)
        vert = vert - center
        scale = np.max(np.abs(vert))
        vert = vert / scale

        dt_masks = np.zeros((100,1280,1280))
        for i, rot in enumerate(np.linspace(-np.pi, np.pi, num=100)):
            pose = np.array([0, rot, 0, 0, 0, 5.5])
            mask = render_car(pose, vert, car_face)
            dt_masks[i] = mask

        for gt_i, gt_car_id in enumerate(gt_car_ids):
            gt_masks = car_vertices[gt_car_id]
            sims = np.zeros(100)
            
            for i, rot in enumerate(np.linspace(0, np.pi, num=100)):
                mask1 = dt_masks[i]
                mask2 = gt_masks[i]
                sims[i] = IOU(mask1, mask2)

            sims_mat[dt_i][gt_i] = np.mean(sims)
    return sims_mat

def pose_similarity(dt, gt, shape_sim_mat, dt_vertices = None, car_vertices = None, car_face = None):
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
    car_num = shape_sim_mat.shape[0]

    dt_car_id = np.uint32(dt[:, -1])
    gt_car_id = np.uint32(gt[:, -1])

    # idx = np.tile(dt_car_id[:, None], (1, gt_num)).flatten() * car_num + \
    #         np.tile(gt_car_id[None, :], (dt_num, 1)).flatten()
    # sims_shape = shape_sim_mat.flatten()[idx]
    # sims_shape = np.reshape(sims_shape, [dt_num, gt_num])
    sims_shape = compute_reproj(car_vertices, car_face, gt_car_id, dt_vertices)
    
    # translation similarity
    dt_car_trans = dt[:, 3:-1]
    gt_car_trans = gt[:, 3:-1]
    dis_trans = np.linalg.norm(np.tile(dt_car_trans[:, None, :], [1, gt_num, 1]) - \
            np.tile(gt_car_trans[None, :, :], [dt_num, 1, 1]), axis=2)

    # rotation similarity
    dt_car_rot = uts.euler_angles_to_quaternions(dt[:, :3])
    gt_car_rot = uts.euler_angles_to_quaternions(gt[:, :3])
    q1 = dt_car_rot / np.linalg.norm(dt_car_rot, axis=1)[:, None]
    q2 = gt_car_rot / np.linalg.norm(gt_car_rot, axis=1)[:, None]


    # diff = abs(np.matmul(q1, np.transpose(q2)))
    diff = abs(1 - np.sum(np.square(np.tile(q1[:, None, :], [1, gt_num, 1]) - \
            np.tile(q2[None, :, :], [dt_num, 1, 1])), axis=2) / 2.0)
    dis_rot = 2 * np.arccos(diff) * 180 / np.pi
    
    return sims_shape, dis_trans, dis_rot



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


