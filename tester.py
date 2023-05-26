import os, json, math

import torch
import cv2
from tqdm import tqdm
import numpy as np

from pytorch3d.io import load_obj

from data_loader  import apollo_3dpose_loader, apollo_eval_loader
import time

opj = os.path.join
def test_model(cfg, model):
    model.eval()
   
    # prepare save folders
    res_path = os.path.join(cfg.OUTPUT_DIR,'res')
    mesh_path = os.path.join(cfg.OUTPUT_DIR,'mesh')

    # save folder setting
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
        
    # Load test dataset
    dataset = apollo_3dpose_loader(cfg.DATASETS.TEST[0], eval=True)
    steps = math.ceil(len(dataset)/cfg.SOLVER.IMS_PER_BATCH)
    dataloader =  apollo_eval_loader(dataset, batch_size = cfg.SOLVER.IMS_PER_BATCH, steps = steps, resize = cfg.DATASETS.RESIZE)

    # model scaling setting it used at model
    model.roi_heads.rx = cfg.DATASETS.RESIZE[1]/dataset[0]['width']
    model.roi_heads.ry = cfg.DATASETS.RESIZE[0]/dataset[0]['height']

    # Inference
    print('Test now begin...')
    thr = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    total_time = 0

    with torch.no_grad():
        for _ in tqdm(range(steps)):
            batch = next(dataloader)
            before = time.time()
            outputs = model(batch)
            total_time += (time.time() - before)
            save_results(batch, outputs, cfg, save_path = (res_path,  mesh_path), thr = thr)


def save_results(inputs, outputs, cfg, save_path = (None,None), thr=0):
    
    res_path, mesh_path = save_path

    if res_path is None:
        res_path=os.path.join(cfg.OUTPUT_DIR,'res')
    if mesh_path is None:
        mesh_path=os.path.join(cfg.OUTPUT_DIR,'mesh')


    for input, output in zip(inputs, outputs):

        # Load inputs
        instances = output['instances']
        
        # Load outputs
        pred_boxes = np.array([b.cpu().detach().numpy() for b in instances.pred_boxes])
        scores = instances.scores.cpu().detach().numpy()
        pred_classes = instances.pred_classes.cpu().detach().numpy() 
        pred_rotates = instances.pred_rotates.cpu().detach().numpy()
        pred_trans = instances.pred_trans.cpu().detach().numpy()
        pred_meshes = instances.pred_meshes.cpu().detach().numpy()
        log_variance = (-(0.5*instances.log_variance).exp()).exp().cpu().detach().numpy()
        pred_keypoints = instances.pred_keypoints.cpu().detach().numpy()

        # Save results to Apollo3D format
        results = []
        meshes = []
        for idx in range(len(instances)):

            score = float(scores[idx]) *log_variance[idx]
            if score < thr:  continue
            
            obj = {}
            rotate = pred_rotates[idx].tolist()
            trans = pred_trans[idx].tolist()
            pred_car = int(pred_classes[idx])
            x1, y1, x2, y2 = pred_boxes[idx]
            keypoint = pred_keypoints[idx]

            obj['pose'] = [float(i) for i in rotate + trans]   
            obj['car_id'] = pred_car
            obj['score'] = score
            obj['bbox'] = [float(i) for i in [x1, y1, x2, y2]]
            obj['visible_rate'] = 1.0        
            obj['keypoints'] = keypoint.reshape(66*3,).tolist()
    
            results.append(obj)
            vert = pred_meshes[idx]
            meshes.append(vert)

        with open(opj(res_path,input['filename']+'.json'), "w") as json_file:
            json.dump(results, json_file)
        
        meshes = np.array(meshes)
        np.save(opj(mesh_path,input['filename']+'.npy'),meshes)
      
