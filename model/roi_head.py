# This code is writen based on Detectron2
import os, json
from collections import OrderedDict
import inspect
from tkinter import X
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import math

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads import ROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY
# from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints

from losses import l1_loss_trans, l2_loss_mesh, l1_loss_rotate, d3d_loss

from detectron2.layers import batched_nms, cat

import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from pytorch3d.io import load_obj
from .nms import nms_3d

class LayerNorm(nn.Module):
    def __init__(self, dim, channel):
        super().__init__()
        self.norm = nn.LayerNorm([channel, dim])

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)
        x = self.norm(x)
        x = x.reshape(B, T, N, C)
        return x

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class RelationAwareAttention(nn.Module):
    def __init__(self, dim, num_head=8):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(-1)
        self.num_head = num_head
        
    def forward(self, x, num_inst_per_image):
        B, C = x.shape

        q_global = self.q(x)
        k_global = self.k(x)
        v_global = self.v(x)

        q_per_img = q_global.split(num_inst_per_image)
        k_per_img = k_global.split(num_inst_per_image)
        v_per_img = v_global.split(num_inst_per_image)

        # self attention per images
        xs = []
        for q, k, v in zip(q_per_img, k_per_img, v_per_img):
            img_B = q.shape[0]
            q = q.reshape(1, img_B, self.num_head, -1).transpose(1, 2)
            k = k.reshape(1, img_B, self.num_head, -1).transpose(1, 2)
            v = v.reshape(1, img_B, self.num_head, -1).transpose(1, 2)

            attn = q @ k.transpose(-1, -2) / np.sqrt(C)
            attn = self.softmax(attn)
            x = (attn @ v).transpose(1, 2).reshape(img_B, C)
            
            xs.append(x)
        
        x = torch.cat(xs,dim=0)
        x = self.u(x)

        return x

class SecneAwareAttention(nn.Module):
    def __init__(self, dim, num_head=8):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(-1)
        self.num_head = num_head
    
    def forward(self, x1, x2, num_inst_per_image,  m2=None):

        q_global = self.q(x1)
        q_per_img = q_global.split(num_inst_per_image)
        k_per_img = self.k(x2)
        v_per_img = self.v(x2)

        xs = []
        for q, k, v in zip(q_per_img, k_per_img, v_per_img):
            B1, C1 = q.shape
            B2, C2 = k.shape

            q = q.reshape(1, B1, self.num_head, -1).transpose(1, 2)
            k = k.reshape(1, B2, self.num_head, -1).transpose(1, 2)
            v = v.reshape(1, B2, self.num_head, -1).transpose(1, 2)

            attn = q @ k.transpose(-1, -2) / np.sqrt(C1)
            attn = self.softmax(attn)
            x = (attn @ v).transpose(1, 2).reshape(B1, C1)
            xs.append(x)
        
        x = torch.cat(xs,dim=0)
        x = self.u(x)
        return x

class BCA(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.sca = RelationAwareAttention(hidden_dim)
        self.gca = SecneAwareAttention(hidden_dim)
        self.mlp = MLP(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm1_1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.scale1 = nn.Parameter(1e-6 * torch.ones(1, hidden_dim), requires_grad=True)
        self.scale2 = nn.Parameter(1e-6 * torch.ones(1, hidden_dim), requires_grad=True)
        self.scale3 = nn.Parameter(1e-6 * torch.ones(1, hidden_dim), requires_grad=True)
       
    def forward(self, x, global_features, num_inst_per_image):
        # Add positial encoding
        x = self.embedding(x)

        # Self-Attention: Relation information
        h1 = self.norm1(x)
        h1 = self.sca(h1, num_inst_per_image)

        # Cross-Attention: Scene information
        h1_1 = self.norm1_1(x)
        h2 = self.norm2(global_features)
        h2 = self.gca(h1_1, h2, num_inst_per_image=num_inst_per_image)
        
        # Fuse features
        x = x + self.scale1 * h1 + + self.scale2 * h2

        # MLP
        h3 = self.norm3(x)
        h3 = self.mlp(h3)

        x = x + self.scale3 * h3
        return x


class ShapeFusion(nn.Module):
    def __init__(self, M, V, input_dim, output_dim=128, num_head = 8, attention_layers=2):
        super(ShapeFusion, self).__init__()
        self.output_dim = output_dim

        self.M = nn.Parameter(M.clone(), requires_grad=False)
        self.V = nn.Parameter(V.clone() - M.clone(), requires_grad=False)
        self.E = nn.Parameter(torch.rand(V.shape[0], input_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.E)

        self.roi_embdedding = nn.Linear(input_dim, output_dim)
        self.q = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.u = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.GELU(),
            nn.Linear(output_dim, self.V.shape[1])
        )

        self.softmax = nn.Softmax(-1)
        self.n_verts = self.M.shape[1]//3

    def forward(self, x):

        N = x.shape[0]
    
        # Q, K, V
        x = self.roi_embdedding(x)
        q = self.q(x)
        k = self.k(self.E) 
        v = self.v(self.E)

        # make attention matrix
        attn_logit = q @ k.transpose(0, 1) / np.sqrt(self.output_dim) 
        attn = self.softmax(attn_logit) #
        x = attn @ v + x

        # mesh
        coarse_mesh = (attn @ self.V)
        coarse_mesh = coarse_mesh + self.u(x)

        return (self.M + coarse_mesh).view(N, self.n_verts, 3)

@ROI_HEADS_REGISTRY.register()
class BAAMROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        input_shape,
        device,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes
        roi_dim = input_shape[box_in_features[0]].channels
        self.device = device
        self._init_pose_head(roi_dim)
        # inference all boxes
        self.box_predictor.test_score_thresh = 0

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        ret.update({'input_shape':input_shape})
        ret['device'] = cfg.MODEL.DEVICE
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def _init_pose_head(self, input_dim):
        
        # set parameters
        self.camera_intrisic = [2304.54786556982, 2305.875668062, 1686.23787612802, 1354.98486439791]
        self.rx = 1873/3384
        self.ry = 1500/2710
        self.num_cars = 79

        # set feature dimension
        self.input_dim = input_dim
        self.hidden_dim = 256
        self.roi_dim = 128
        self.num_global_context = 8
        self.heatmap_dim = 66
        self.keypoint_dim = 300

        self._global_features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.input_dim, self.hidden_dim, 3, 1, 1)),
            ('norm1', nn.GroupNorm(16, self.hidden_dim)),
            ('act1', nn.GELU()),
            ('conv2', nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1)),
            ('norm2', nn.GroupNorm(16, self.hidden_dim)),
            ('act2', nn.GELU()),
            ('conv3', nn.Conv2d(self.hidden_dim, self.num_global_context * self.hidden_dim, 1, 1)),
            ('pool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten())
        ]))

        # box feature extractor
        self._roi_features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.input_dim, self.hidden_dim, 3, 2, 1)),
            ('norm1', nn.GroupNorm(16, self.hidden_dim)),
            ('act1', nn.GELU()),
            ('conv2', nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 2, 1)),
            ('norm2', nn.GroupNorm(16, self.hidden_dim)),
            ('act2', nn.GELU()),
            ('conv3', nn.Conv2d(self.hidden_dim, self.roi_dim, 3, 2, 1)),
            ('flatten', nn.Flatten())
        ]))
        # box extractor
        self._box_pos = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4, self.roi_dim)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(self.roi_dim, self.roi_dim)),
        ]))

        # keypoint extactor
        self._keypoint_pos = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(66*2, self.keypoint_dim)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(self.keypoint_dim, self.keypoint_dim)),
        ]))
        
        # keypoint visibility extractor
        self._keypoint_weights = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(66, self.keypoint_dim)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(self.keypoint_dim, self.keypoint_dim)),
        ]))

        # rotation feature extractor
        self.rotate_feature = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.roi_dim + self.roi_dim + self.keypoint_dim, self.hidden_dim)),
            ('norm1', nn.GroupNorm(16, self.hidden_dim)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(self.hidden_dim, self.hidden_dim)),
        ]))

        # translation feature extractor
        self.trans_feature = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.roi_dim + self.roi_dim + self.keypoint_dim, self.hidden_dim)),
            ('norm1', nn.GroupNorm(16, self.hidden_dim)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(self.hidden_dim, self.hidden_dim)),
        ]))

        # shape feature extractor
        self.shape_feature = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.roi_dim + self.roi_dim + self.keypoint_dim, self.hidden_dim)),
            ('norm1', nn.GroupNorm(16, self.hidden_dim)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(self.hidden_dim, self.hidden_dim)),
        ]))
       
        self._context_fusion = BCA(self.hidden_dim, self.hidden_dim)
        self._regress_translation = nn.Linear(self.hidden_dim, 4)
        self._regress_rotation = nn.Linear(self.hidden_dim, 3)


        # init meshes
        template_path = 'car_template/'
        optim = True

        self.init_V = torch.from_numpy(np.load(template_path + 'V.npy').astype('float32')).to(self.device)
        self.init_M = torch.from_numpy(np.load(template_path + 'M.npy').astype('float32')).to(self.device)
        self.n_verts = self.init_M.shape[1]//3

        self._shape_fusion = ShapeFusion(self.init_M, self.init_V, self.hidden_dim, self.hidden_dim, num_head = 8, attention_layers=2)
    
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                weight_init.c2_msra_fill(m)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        weight_init.c2_xavier_fill(self._regress_translation)
        weight_init.c2_xavier_fill(self._regress_rotation)        


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        train_3d = True,
        train_key = True
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            proposals = add_ground_truth_to_proposals(targets, proposals)
        del targets

        if self.training:
            losses = {}
            # # add 2D Head Loss
            box_losses = self._forward_box(features, proposals)
            losses.update(box_losses)
            if train_key :
                kpt_losses, proposals = self._forward_keypoint(features, proposals)
                losses.update(kpt_losses)
            if train_3d:
                d3d_losses = self._forward_3d_pose(features, proposals)
                losses.update(d3d_losses)
            return losses

        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        
        _, instances = self._forward_keypoint(features, instances)
        instances = self._forward_3d_pose(features,  instances)
        return instances

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            with torch.no_grad():
                pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                    predictions, proposals
                )
                # compute pred classes
                num_inst_per_image = [len(p) for p in proposals]
                pred_classes = torch.argmax(predictions[0][:,:-1],dim=1)
                pred_classes = pred_classes.split(num_inst_per_image, dim=0)
                probs = F.softmax(predictions[0], dim=-1)
                probs = probs[:,:-1].split(num_inst_per_image, dim=0)
                # Add pred information to proposals
                for p, pred_boxe, pred_class, prob in zip(proposals, pred_boxes, pred_classes, probs):
                    p.pred_boxes = Boxes(pred_boxe)
                    p.pred_classes = pred_class
                    p.probs = prob
            
            return losses
        else:
            pred_instances, inds = self.box_inference(predictions, proposals)
            num_inst_per_image = [len(p) for p in proposals]
            probs = F.softmax(predictions[0], dim=-1)
            probs = probs[:,:-1].split(num_inst_per_image, dim=0)
            for idx, pred in enumerate(pred_instances):
                pred.proposal_boxes = Boxes(proposals[idx].proposal_boxes.tensor[inds[idx]])
                pred.probs = probs[idx][inds[idx]]
            return pred_instances


    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """

        
        features = [features[f] for f in self.keypoint_in_features]

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances_vis = select_proposals_with_visible_keypoints(instances)
            boxes_vis = [x.proposal_boxes for x in instances_vis]

            features_vis = self.keypoint_pooler(features, boxes_vis)
            loss = self.keypoint_head(features_vis, instances_vis)
        else:
            loss = {}

        boxes = [x.pred_boxes  for x in instances]
        features = self.keypoint_pooler(features, boxes)

        self.keypoint_head.eval()
        with torch.no_grad():
            instances = self.keypoint_head(features, instances)
        self.keypoint_head.train()

        return loss, instances


    def _forward_3d_pose(self, features, proposals):
        
        if self.training:      
            # call GT label
            gt_rotates = torch.cat([p.gt_rotate for p in proposals], dim=0)
            gt_trans = torch.cat([p.gt_trans for p in proposals], dim=0)
            gt_verts = torch.cat([p.gt_verts for p in proposals], dim=0)

        # global features
        global_features = features['p7']
        global_features = self._global_features(global_features)
        global_features = global_features.reshape(len(proposals), self.num_global_context, -1)

        # roi features
        roi_features = [features[f] for f in self.box_in_features]
        roi_features = self.box_pooler(roi_features, [x.pred_boxes for x in proposals])
        roi_features = self._roi_features(roi_features)

        # box coord features
        box_pos = torch.cat([p.pred_boxes.tensor for p in proposals], dim=0)
        new_box_pos = box_pos.clone()
        # normalize box coord
        new_box_pos[:,0] = (0.5 * (box_pos[:,0] + box_pos[:,2]) /self.rx - self.camera_intrisic[2]) / self.camera_intrisic[0]
        new_box_pos[:,1] = (0.5 * (box_pos[:,1] + box_pos[:,3]) / self.ry - self.camera_intrisic[3]) / self.camera_intrisic[1]
        new_box_pos[:,2] = (box_pos[:,2] - box_pos[:,0]) / self.rx / self.camera_intrisic[0]
        new_box_pos[:,3] = (box_pos[:,3] - box_pos[:,1]) / self.ry / self.camera_intrisic[1]
        box_pos_features = self._box_pos(new_box_pos)

        # keypoint coord features
        keypoint_pos = torch.cat([p.pred_keypoints for p in proposals], dim=0)
        new_keypoint_pos = keypoint_pos.clone()
        # nomarlize keypoint coord
        new_keypoint_pos[:,:,0] = (keypoint_pos[:,:,0]/self.rx - self.camera_intrisic[2]) / self.camera_intrisic[0]
        new_keypoint_pos[:,:,1] = (keypoint_pos[:,:,1]/self.ry - self.camera_intrisic[3]) / self.camera_intrisic[1]
        # multiply visibility
        keypoints_vis = new_keypoint_pos[:,:,2]
        new_keypoint_pos = new_keypoint_pos[:,:,:2].reshape(keypoint_pos.shape[0], -1)
        keypoint_pos_features = self._keypoint_pos(new_keypoint_pos)
        keypoints_weights = self._keypoint_weights(keypoints_vis)
        keypoint_pos_features *= keypoints_weights.sigmoid()

        # object featueres
        object_features = torch.cat([roi_features, box_pos_features, keypoint_pos_features], dim= 1)

        # predict translation
        num_inst_per_image = [len(p) for p in proposals]
        translation_feature = self.trans_feature(object_features)
        translation_feature = self._context_fusion(translation_feature, global_features,num_inst_per_image)
        pred_trans = self._regress_translation(translation_feature)

        # predict rotation
        rotation_feature = self.rotate_feature(object_features)
        pred_rotates = self._regress_rotation(rotation_feature)

        # predict shape
        shape_feature = self.shape_feature(object_features)
        pred_verts = self._shape_fusion(shape_feature)
        pred_verts = (pred_verts).view(pred_verts.shape[0], self.n_verts, 3)

        if self.training:

            # compute losses
            loss_trans = l1_loss_trans(pred_trans, gt_trans)
            loss_rotate = l1_loss_rotate(pred_rotates, gt_rotates)
            loss_mesh = l2_loss_mesh(pred_verts, gt_verts)

            loss_RT, loss_R, loss_T = d3d_loss(
                (gt_verts, gt_trans, gt_rotates),
                (pred_verts, pred_trans, pred_rotates)
            )
            return {
                'loss_rotate': loss_rotate, 'loss_trans': loss_trans,
                'loss_mesh' : loss_mesh, 
                'loss_R': loss_R, 'loss_T': loss_T, 'loss_RT': loss_RT,
                }
        else:

            i = 0
            for proposal in proposals:
                proposal.pred_trans = pred_trans[:,:3][i:i+len(proposal)] 
                proposal.log_variance = pred_trans[:,3][i:i+len(proposal)] 
                proposal.pred_rotates = pred_rotates[i:i+len(proposal)] 
                proposal.pred_meshes = pred_verts[i:i+len(proposal)] 
                i += len(proposal)
            
            # 3D NMS
            if not hasattr(proposals[0],'gt_classes'):
                proposals = nms_3d(proposals)
            return proposals


    def box_inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.box_predictor.predict_boxes(predictions, proposals)
        scores = self.box_predictor.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, 
            self.box_predictor.test_score_thresh, self.box_predictor.test_nms_thresh, self.box_predictor.test_topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
         ]

        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]



def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    # boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # NMS class dependently
    dummy_inds = torch.zeros_like(filter_inds[:, 1])
    # 2. Apply NMS for each class independently.
    # keep = batched_nms(boxes, scores, dummy_inds, nms_thresh)
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]