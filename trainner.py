import logging
import os, math
import time

import torch
from torch import nn
from tqdm import tqdm

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
from detectron2.engine import default_writers
from detectron2.data.samplers import TrainingSampler
from detectron2.solver import build_lr_scheduler, build_optimizer
from fvcore.common.timer import Timer

from data_loader  import apollo_mapper, apollo_3dpose_loader
from tester import save_results
from pytorch3d.io import  load_obj
import torch.optim as optim
import numpy as np

import cv2

p3d_weights = {
    'loss_keypoint':0.1,
    'loss_rotate': 1, 'loss_trans': 0.5, 'loss_mesh':3, 
    'loss_R': 0.1, 'loss_T': 0.01,  'loss_RT': 0.01
    }


def train_model(cfg, model, eval = False):

    model.train()

    # Load optimizer
    optimizer = optim.AdamW(model.parameters(), lr = cfg.SOLVER.BASE_LR)

    # Load logger
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    checkpointer.save('res2net_bifpn_light.pth')

    # Load train data
    dataset = apollo_3dpose_loader(cfg.DATASETS.TRAIN[0])
    data_loader = build_detection_train_loader(
        dataset, 
        mapper = apollo_mapper(cfg.DATASETS.RESIZE),
        sampler=TrainingSampler(len(dataset)),
        total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS
    )

    # set rescaling factor. BAAM uses them at 3d pose estimation.
    model.roi_heads.rx = cfg.DATASETS.RESIZE[1]/dataset[0]['width']
    model.roi_heads.ry = cfg.DATASETS.RESIZE[0]/dataset[0]['height']

    # Train setting
    epoch_iter = math.ceil((len(dataset)/cfg.SOLVER.IMS_PER_BATCH))

    print("{} iters per epoch".format(epoch_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):

            # pre-time check 
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()

            if  iteration == cfg.SOLVER.STEPS[0]:
                for g in optimizer.param_groups : g ['lr'] *= 0.1


            # calculate losses
            loss_dict = model(data)
            # mulyiply balancing parameters
            
            for k in loss_dict.keys():
                if k in p3d_weights.keys():
                    loss_dict[k] *= p3d_weights[k]  

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # recode losses
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            #recode learning rate
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False) 

            #check post-time
            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 10 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            
