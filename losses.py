from fvcore.nn import smooth_l1_loss
import numpy as np
import torch
import math

def l1_loss_trans(input,target, reduction="mean"):

    target_xy = target[:,:2]
    pred_xy = input[:,:2]
    loss_xy = torch.norm(pred_xy - target_xy, p=1, dim=1)

    target_z = target[:,2]
    pred_z = input[:,2]
    log_variance = input[:,3]

    loss_z = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(pred_z - target_z) + 0.5*log_variance

    
    if reduction == 'mean':
        loss = loss_xy.mean()*2/3 +loss_z.mean()/3
    if reduction == 'sum':
        loss = loss_xy.sum()*2/3 +loss_z.sum()/3
    return loss


def l1_loss_rotate(input, target ,reduction ='mean'):
    
    N = input.shape[0]
    l1_loss = torch.abs(input - target)

    outpi_inds = (l1_loss > math.pi)
    inpi_inds = (l1_loss <= math.pi)

    outpi_loss = torch.abs(2*math.pi - l1_loss[outpi_inds]).sum()
    inpi_loss = l1_loss[inpi_inds].sum()


    if reduction == 'mean':
        return (outpi_loss + inpi_loss) / N
    if reduction == 'sum':
        return outpi_loss + inpi_loss


    
def l2_loss_mesh(pred_verts, target, reduction = 'mean'):

    n_verts = target.shape[1]
    main_loss = torch.norm(pred_verts-target, dim=2).mean(1)

    if reduction == 'mean':
        return main_loss.mean() 
    if reduction == 'sum':
        return main_loss.sum() 


def d3d_loss(gt, pred):

    device = pred[0].device
    B, N, _ = pred[0].shape

    gt_verts, gt_trans, gt_rotates = gt
    pred_verts, pred_trans, pred_rotates = pred

    # make pred and gt R matrix 
    pred_R = euler_angles_to_rotation_matrix(pred_rotates).to(device)
    gt_R = euler_angles_to_rotation_matrix(gt_rotates).to(device)

    # prepare verts to employ RT
    pred_verts = pred_verts.transpose(2,1)
    gt_verts  = gt_verts.transpose(2,1)

    # employ R
    pred_R_verts = torch.bmm(pred_R, pred_verts.clone())
    gt_R_verts = torch.bmm(gt_R, gt_verts.clone())

    # compute R_loss
    R_loss = torch.norm(pred_R_verts-gt_R_verts, dim=1).mean(1)
    
    # Employ T
    pred_T_verts = pred_verts.clone() + pred_trans[:,:3].unsqueeze(2)
    gt_T_verts = gt_verts.clone() + gt_trans.unsqueeze(2)

    # comput T_loss
    T_loss = torch.norm(pred_T_verts-gt_T_verts, dim=1).mean(1)

    # emply R, T
    pred_RT_verts = pred_R_verts + pred_trans[:,:3].unsqueeze(2)
    gt_RT_verts = gt_R_verts + gt_trans.unsqueeze(2)

    # comput RT_loss
    RT_loss = torch.norm(pred_RT_verts-gt_RT_verts, dim=1).mean(1)

    return RT_loss.mean(), R_loss.mean(), T_loss.mean()


def euler_angles_to_rotation_matrix(car_rotation, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = car_rotation[:,0], car_rotation[:,1], car_rotation[:,2]

    rollMatrix = torch.tensor([[
        [1, 0, 0],
        [0, math.cos(roll[i]), -math.sin(roll[i])],
        [0, math.sin(roll[i]), math.cos(roll[i])]] for i in range(car_rotation.shape[0])])
    
    
    pitchMatrix = torch.tensor([[
        [math.cos(pitch[i]), 0, math.sin(pitch[i])],
        [0, 1, 0],
        [-math.sin(pitch[i]), 0, math.cos(pitch[i])]] for i in range(car_rotation.shape[0])])

    yawMatrix = torch.tensor([[
        [math.cos(yaw[i]), -math.sin(yaw[i]), 0],
        [math.sin(yaw[i]), math.cos(yaw[i]), 0],
        [0, 0, 1]] for i in range(car_rotation.shape[0])])

    R = torch.matmul(torch.matmul(yawMatrix, pitchMatrix), rollMatrix)
    return R
