import torch
from detectron2.structures import Boxes, Instances

def nms_3d(proposals):
    return [nms_single_img(p) for p in proposals]


def nms_single_img(proposal):
    x_thr = 2.0
    z_thr = 1.5
    maxDet = 100

    # extract score
    scores = proposal.scores.clone().detach()
    log_variance = (-(0.5*proposal.log_variance).exp()).exp()
    # compute 3d score
    scores *= log_variance
    scores, indices = torch.sort(scores, descending=True)
    # extract translation and sorted by 3d score
    pred_trans = proposal.pred_trans.clone().detach()[indices]

    # make mask
    mask = torch.tensor([True] * len(scores))
    
    for idx in range(len(scores)):
        # contunue, if mask is excluded
        if not mask[idx]:
            continue
        
        # compute z, x distance
        trans = pred_trans[idx]
        x_dist = torch.abs(trans[0] -  pred_trans[:,0])
        x_dist[idx] = 999
        
        z_dist = torch.abs(trans[2] -  pred_trans[:,2])
        z_dist[idx] = 999

        filter_inds = z_dist < z_thr
        filter_inds = torch.logical_and(x_dist < x_thr, z_dist < z_thr)
        # exclude supression instances
        mask[filter_inds] = False

    # make new instance
    result = Instances(proposal.image_size)
    result.pred_boxes = Boxes(proposal.pred_boxes.tensor[indices][mask][:maxDet])
    result.scores = proposal.scores[indices][mask][:maxDet]
    result.pred_classes = proposal.pred_classes[indices][mask][:maxDet]
    result.pred_rotates = proposal.pred_rotates[indices][mask][:maxDet]
    result.pred_trans = proposal.pred_trans[indices][mask][:maxDet]
    result.pred_meshes = proposal.pred_meshes[indices][mask][:maxDet]
    result.log_variance = proposal.log_variance[indices][mask][:maxDet]
    result.pred_keypoints= proposal.pred_keypoints[indices][mask][:maxDet]

    return result