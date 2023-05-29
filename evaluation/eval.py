"""
    Brief: Evaluation of 3D car instance mean AP following coco eval criteria
    Author: wangpeng54@baidu.com, (Piotr Dollar and Tsung-Yi Lin)
    Date: 2018/6/20
"""

import os
import json
import numpy as np
import datetime
import argparse
import time
import copy
import utils.eval_utils as euts
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
import pickle as pkl
from tqdm import tqdm
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, look_at_view_transform,
    MeshRasterizer, BlendParams, SoftSilhouetteShader, TexturesVertex
)
import torch
from pytorch3d.structures import Meshes
opj = os.path.join


Criterion = namedtuple('Criterion', [
    'shapeSim',  # Criterion for shape similarity
    'transDis',  # thresholds for translation
    'oriDis',    # thresholds for orientation
    ])


class Evalconfig():
    def __init__(self,cfg):
        self.test_dir = None
        self.gt_dir = cfg.DATASETS.TEST[0] + '/car_poses_labels'
        self.simType = None

class Detect3DEval(object):
    # Interface for evaluating detection on the Apolloscape 3d car understanding
    #
    # The usage for Detection3DEval is as follows:
    #  E = Detect3DEval(args);      # initialize object
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  image_names     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  shapeThrs  - [.5:.05:.95] T=10 shape thresholds for evaluation
    #  rotThrs    - [50:  5:  5] T=10 rot thresholds for evaluation
    #  transThrs  - [0.1:.3:2.8] T=10 trans thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting

    def __init__(self, args):
        """ Initialize CocoEval using coco APIs for gt and dt
        Input:
            args: configeration object containing test folder and gt folder
        """
        if not args.simType:
            args.simType = '3dpose'
            print('simType not specified. use default simType ')

        self.args = args
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.evalRes = {}               # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.sims = {}                      # sims between all gts and dts

        self.car_model_dir = '/data/apollo/car_models'
        self.image_list = self._checker(args.test_dir, args.gt_dir)
        self.params_abs = Params(simType=args.simType)
        self.params_rel = Params(simType=args.simType,mode='rel')
    def _checker(self, res_folder, gt_folder):
        """Check whether results folder contain same image number
        """

        gt_list = sorted(os.listdir(gt_folder))
        res_list = sorted(os.listdir(res_folder))

        if len(gt_list) != len(res_list):
            raise Exception('results folder image num is not the same as ground truth')

        return gt_list

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        base_path = '/'.join(self.args.test_dir.split('/')[:-1])
        mesh_path = opj(base_path,'mesh')

        self._gts = {}       # gt for evaluation
        self._dts = {}       # dt for evaluation
        self._dts_vert = {}
        #print('loading results')
        count_gt = 1
        count_dt = 1
        for image_name in self.image_list:
            gt_file = '%s/%s' % (self.args.gt_dir, image_name)
            dt_file = '%s/%s' % (self.args.test_dir, image_name)
            mesh_file = '%s/%s' % (mesh_path, image_name.rstrip('.json')+'.npy')

            car_poses_gt = json.load(open(gt_file, 'r'))
            car_poses_dt = json.load(open(dt_file, 'r'))
            car_mesh_dt = np.load(mesh_file)
            for car_pose in car_poses_gt:
                car_pose['id'] = count_gt
                car_pose['ignore'] = 0
                count_gt += 1

            for car_pose in car_poses_dt:
                car_pose['id'] = count_dt
                count_dt += 1

            self._gts[image_name] = car_poses_gt
            self._dts[image_name] = car_poses_dt
            self._dts_vert[image_name] = car_mesh_dt

        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.evalRes  = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        #print('Running per image evaluation...')

        self._prepare()
        compute_sim = self.compute_sim
        
        total_images = len(self.image_list)
        self.sims = {}
        for n, image_name in enumerate(self.image_list,start=1):
            print('Evaluate {0} .....({1}/{2})'.format(image_name, n, total_images))
            abs, rel = compute_sim(image_name)
            self.sims[image_name] = {'abs':abs, 'rel':rel}
  
        maxDet = self.params_abs.maxDets[-1]
        self.evalImgs = [self.evaluate_image(image_name, maxDet)
                 for image_name in self.image_list
             ]
        self._paramsEval = {'abs': copy.deepcopy(self.params_abs), 'rel': copy.deepcopy(self.params_rel)}
        toc = time.time()

    def compute_sim(self, image_name):
        """Compute similarity for an image between ground truth and detected results
        """
        gt = self._gts[image_name]
        dt = self._dts[image_name]

        if len(gt) == 0 or len(dt) == 0:
            return []

        inds = np.argsort([-d['score'] for d in dt], kind = 'mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > self.params_abs.maxDets[-1]:
            dt = dt[0:self.params_abs.maxDets[-1]]

        g = np.array([g['pose'] + [g['car_id']] for g in gt], dtype=np.float32)
        d = np.array([d['pose'] + [d['car_id']] for d in dt], dtype=np.float32)

        dt_vertices = self._dts_vert[image_name]
        # compute iou between each dt and gt region
        sims = euts.pose_similarity(d, g, dt_vertices, self.car_models, self.car_models_face, args.light)
        abs_sims = np.stack([sims[0],sims[2],sims[1]], axis=2)
        rel_sims = np.stack([sims[0],sims[3],sims[1]], axis=2)
        return abs_sims, rel_sims


    def evaluate_image(self, image_name, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''

        def _satisfy(sim, cri):
            return sim[0] >= cri.shapeSim \
                   and sim[1] <= cri.transDis \
                   and sim[2] <= cri.oriDis

        abs_p = self.params_abs
        rel_p = self.params_rel
        gt = self._gts[image_name]
        dt = self._dts[image_name]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]

        # load computed sims
        abs_sims = self.sims[image_name]['abs'][:, gtind, :] if len(
                    self.sims[image_name]['abs']) > 0 \
                    else self.sims[image_name]['abs']
        rel_sims = self.sims[image_name]['rel'][:, gtind, :] if len(
                    self.sims[image_name]['rel']) > 0 \
                    else self.sims[image_name]['rel']
        # number of criterion
        T = len(abs_p.shapeThrs)
        G = len(gt)
        D = len(dt)

        abs_gtm  = np.zeros((T, G)) # match of gt
        abs_dtm  = np.zeros((T, D)) # match of detection
        rel_gtm  = np.zeros((T, G)) # match of gt
        rel_dtm  = np.zeros((T, D)) # match of detection
        gtIg = np.array([g['_ignore'] for g in gt]) # ignore gt index
        dtIg = np.zeros((T, D)) # detection ignore

        # finding matches between detections & ground truth
        if not len(abs_sims) == 0:
            for tind, (abs_cri,rel_cri) in enumerate(zip(abs_p.criteria, rel_p.criteria)):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    cur_abs_cri = abs_cri
                    cur_rel_cri = rel_cri
                    abs_m   = -1
                    rel_m   = -1
                    for gind, g in enumerate(gt):

                        # if match successful and best so far, store appropriately
                        if _satisfy(abs_sims[dind, gind],  cur_abs_cri):
                            if not abs_gtm[tind, gind]>0:
                                cur_abs_match = abs_sims[dind, gind]
                                cur_abs_cri = Criterion(cur_abs_match[0], cur_abs_match[1], cur_abs_match[2])
                                abs_m = gind
                        if _satisfy(rel_sims[dind, gind],  cur_rel_cri):
                            if not rel_gtm[tind, gind]>0:
                                cur_rel_match = rel_sims[dind, gind]
                                cur_rel_cri = Criterion(cur_rel_match[0], cur_rel_match[1], cur_rel_match[2])
                                rel_m = gind

                    # if match made store id of match for both dt and gt
                    if abs_m != -1:
                        abs_gtm[tind, abs_m]     = d['id']
                        abs_dtm[tind, dind]  = gt[abs_m]['id']
                    if rel_m != -1:
                        rel_gtm[tind, rel_m]     = d['id']
                        rel_dtm[tind, dind]  = gt[rel_m]['id']

        a = np.array([False for d in dt]).reshape((1, len(dt)))

        # store results for given image and category
        return {
                'image_id':     image_name,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'absMatches':   [abs_gtm, abs_dtm],
                'relMatches':   [rel_gtm, rel_dtm],
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
            }

    def accumulate(self, p = None):
        ''' Accumulate per image evaluation results and store the result in self.eval

        Inputs:
            p: input params for evaluation
        '''

        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            abs_p = self.params_abs
            rel_p = self.params_rel

        abs_p.catIds = [-1]
        rel_p.catIds = [-1]
        T           = len(abs_p.simThrs)                       # number of thresh
        R           = len(abs_p.recThrs)                       # number of recall thresh
        abs_recall      = -np.ones((T))
        rel_recall      = -np.ones((T))
        abs_scores      = -np.ones((T, R))
        abs_precision   = -np.ones((T, R)) # -1 for the precision of absent categories
        rel_scores      = -np.ones((T, R))
        rel_precision   = -np.ones((T, R)) # -1 for the precision of absent categories
        # create dictionary for future indexing
        # _pe = self._paramsEval
        # catIds = [-1]
        # setK = set(catIds)
        # setA = set(map(tuple, _pe.areaRng))
        # setM = set(_pe.maxDets)
        setI = set(range(len(self.image_list))) # can use to select image ids

        # get inds to evaluate
        # k_list = [n for n, k in enumerate(abs_p.catIds)  if k in setK]
        # m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        # a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(range(len(self.image_list)))  if i in setI]
        # A0 = len(_pe.areaRng)
        # I0 = len(_pe.image_names)

        # retrieve E at each category, area range, and max number of detections
        E = [self.evalImgs[i] for i in i_list]
        E = [e for e in E if not e is None]

        dtScores = np.concatenate([e['dtScores'][0:abs_p.maxDets[-1]] for e in E])

        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        inds = np.argsort(-dtScores, kind='mergesort')
        dtScoresSorted = dtScores[inds]

        abs_dtm  = np.concatenate([e['absMatches'][1][:,0:abs_p.maxDets[-1]] for e in E], axis=1)[:, inds]
        rel_dtm  = np.concatenate([e['relMatches'][1][:,0:abs_p.maxDets[-1]] for e in E], axis=1)[:, inds]
        # dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:, inds]
        gtIg = np.concatenate([e['gtIgnore'] for e in E])
        npig = np.count_nonzero(gtIg==0)

        abs_tps = abs_dtm != 0
        abs_fps = abs_dtm == 0
        rel_tps = rel_dtm != 0
        rel_fps = rel_dtm == 0

        abs_tp_sum = np.cumsum(abs_tps, axis=1).astype(dtype=np.float)
        abs_fp_sum = np.cumsum(abs_fps, axis=1).astype(dtype=np.float)
        rel_tp_sum = np.cumsum(rel_tps, axis=1).astype(dtype=np.float)
        rel_fp_sum = np.cumsum(rel_fps, axis=1).astype(dtype=np.float)

        for t, (abs_tp, abs_fp, rel_tp, rel_fp) in enumerate(zip(abs_tp_sum, abs_fp_sum, rel_tp_sum, rel_fp_sum)):
            abs_tp = np.array(abs_tp)
            abs_fp = np.array(abs_fp)

            rel_tp = np.array(rel_tp)
            rel_fp = np.array(rel_fp)


            abs_nd = len(abs_tp)
            rel_nd = len(rel_tp)
            abs_rc = abs_tp / npig
            rel_rc = rel_tp / npig

            abs_pr = abs_tp / (abs_fp + abs_tp + np.spacing(1))
            rel_pr = rel_tp / (rel_fp + rel_tp + np.spacing(1))
            
            abs_q  = np.zeros((R, ))
            abs_ss = np.zeros((R, ))
            rel_q  = np.zeros((R, ))
            rel_ss = np.zeros((R, ))

            if abs_nd:
                abs_recall[t] = abs_rc[-1]
            else:
                recall[t] = 0
            if rel_nd:
                rel_recall[t] = rel_rc[-1]
            else:
                rel_recall[t] = 0

            # numpy is slow without cython optimization for accessing elements
            # use python array gets significant speed improvement
            abs_pr = abs_pr.tolist(); abs_q = abs_q.tolist()
            for i in range(abs_nd-1, 0, -1):
                if abs_pr[i] > abs_pr[i-1]:
                    abs_pr[i-1] = abs_pr[i]
            rel_pr = rel_pr.tolist(); rel_q = rel_q.tolist()
            for i in range(rel_nd-1, 0, -1):
                if rel_pr[i] > rel_pr[i-1]:
                    rel_pr[i-1] = rel_pr[i]

            abs_inds = np.searchsorted(abs_rc, abs_p.recThrs, side='left')
            rel_inds = np.searchsorted(rel_rc, abs_p.recThrs, side='left')
            try:
                for ri, pi in enumerate(abs_inds):
                    abs_q[ri] = abs_pr[pi]
                    abs_ss[ri] = dtScoresSorted[pi]
            except:
                pass
            try:
                for ri, pi in enumerate(rel_inds):
                    rel_q[ri] = rel_pr[pi]
                    rel_ss[ri] = dtScoresSorted[pi]
            except:
                pass

            abs_precision[t, :] = np.array(abs_q)
            abs_scores[t, :] = np.array(abs_ss)

            rel_precision[t, :] = np.array(rel_q)
            rel_scores[t, :] = np.array(rel_ss)

        self.abs_evalRes = {
            'params': abs_p,
            'counts': [T, R],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': abs_precision,
            'recall':   abs_recall,
            'scores': abs_scores,
        }
        self.rel_evalRes = {
            'params': rel_p,
            'counts': [T, R],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': rel_precision,
            'recall':   rel_recall,
            'scores': rel_scores,
        }
        toc = time.time()

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap=1, simThr=None, maxDets=100):
            """Summarize an evaluation
            Input:
                ap: whether ask for average precision (ap = 1) or average recall.
                simThr: which sim criterion is considered.
                areaRng: which area range is considered.
                maxDets: max  number of detection per image
                f: the file handler for output stream

            """
            abs_p = self.params_abs
            rel_p = self.params_rel

            iStr = ' {:<18} {} @[ Criteria={:<9}] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            simstr = 'c0:c10' if simThr is None else simThr

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                abs_s = self.abs_evalRes['precision']
                rel_s = self.rel_evalRes['precision']
                # sim matrix
                if simThr is not None:
                    t = [abs_p.simThrs.index(simThr)]
                    abs_s = abs_s[t]
                    rel_s = rel_s[t]

            else:
                # dimension of recall: [TxKxAxM]
                abs_s = self.abs_evalRes['recall']
                rel_s = self.rel_evalRes['recall']
                if simThr is not None:
                    t = abs_p.simThrs.index(simThr)
                    abs_s = abs_s[t]
                    rel_s = rel_s[t]

            if len(abs_s[abs_s>-1])==0:
                abs_mean_s = -1
            else:
                abs_mean_s = np.mean(abs_s[abs_s>-1])

            if len(rel_s[rel_s>-1])==0:
                rel_mean_s = -1
            else:
                rel_mean_s = np.mean(rel_s[rel_s>-1])
            return abs_mean_s, rel_mean_s

        def _summarizeDets():

            abs_stats = np.zeros(11)
            rel_stats = np.zeros(11)
            abs_stats[0], rel_stats[0] = _summarize(1)
            abs_stats[1], rel_stats[1] = _summarize(1, simThr='c0')
            abs_stats[2], rel_stats[2] = _summarize(1, simThr='c1')
            abs_stats[3], rel_stats[3] = _summarize(1, simThr='c2')
            abs_stats[4], rel_stats[4] = _summarize(1, simThr='c3')
            abs_stats[5], rel_stats[5] = _summarize(1, simThr='c4')
            abs_stats[6], rel_stats[6] = _summarize(1, simThr='c5')
            abs_stats[7], rel_stats[7] = _summarize(1, simThr='c6')
            abs_stats[8], rel_stats[8] = _summarize(1, simThr='c7')
            abs_stats[9], rel_stats[9] = _summarize(1, simThr='c8')
            abs_stats[10], rel_stats[10] = _summarize(1, simThr='c9')

            return abs_stats, rel_stats

        if not self.abs_evalRes:
            raise Exception('Please run accumulate() first')
        summarize = _summarizeDets
        abs_stats, rel_stats = summarize()

        print(abs_stats)
        print(rel_stats)

        metric_names = ['mean'] + [f'c{i}' for i in range(10)]
        f = open(self.args.res_file, 'w')
        f.write('A3DP-Abs\n')
        for name, value in zip(metric_names, abs_stats):
            f.write('%s %.4f\n' % (name, value))
        f.write('A3DP-Rel\n')
        for name, value in zip(metric_names, rel_stats):
            f.write('%s %.4f\n' % (name, value))
        f.close()

    def __str__(self):
        self.summarize()

    def load_car_models(self):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        print('Load car models....(it trakes some minutes)')
        ids = json.load(open('id_to_abb.json'))

        for model in tqdm(ids.keys()):

            if args.light:
                # Load mask directly
                masks = np.zeros((10,128,128))
                for i in range(0,10):
                    mask = cv2.imread('rot10/{0}rot{1}.png'.format(model,i))
                    mask = mask[...,0]/255
                    masks[i] = mask
                self.car_models[int(model)] = masks
            else:
                masks = np.zeros((100,1280,1280))
                for i in range(0,100):
                    mask = cv2.imread('rot100/{0}rot{1}.png'.format(model,i))
                    mask = mask[...,0]/255
                    masks[i] = mask
                self.car_models[int(model)] = masks

        gsnet_mesh = load_obj('../apollo_deform/0.obj')
        self.car_models_face = gsnet_mesh[1].verts_idx.numpy()

    def __str__(self):
        self.summarize()


class Params(object):
    """Params for apolloscape 3d car instance evaluation api
    Inputs:
        simType: currently only '3dpose' is supported. Later we may add '3dbbox' for eval
    """
    def __init__(self, simType='3dpose', mode='abs'):

        self.shapeThrs = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)
        self.oriThrs = np.linspace(30, 3, round((30 - 3) / 3) + 1, endpoint=True)
        if mode == 'abs':
            self.transThrs = np.linspace(2.8, 0.1, round((2.8 - 0.1) / .3) + 1, endpoint=True)
        else:
            self.transThrs = np.linspace(0.1, 0.01, round((0.1 - 0.01) / 0.01) + 1, endpoint=True)

        assert mode in ['abs','rel'], 'mode should be abs or rel'
        if simType == '3dpose':
            self.set_det_params()
        else:
            raise Exception('simType not supported')
        self.simType = simType

    def set_det_params(self):
        self.image_names = []
        self.catIds = []

        script_path = os.path.abspath(os.path.dirname(__file__))

        self.criterion_num = len(self.shapeThrs)
        self.simThrs = ['c' + str(i) for i in range(self.criterion_num)]

        assert self.criterion_num == len(self.oriThrs)
        assert self.criterion_num == len(self.transThrs)

        self.recThrs = np.linspace(.0, 1.00, round((1.00 - .0) / .01) + 1, endpoint=True)
        # from loss to strict criterion
        self.criteria = [Criterion(self.shapeThrs[i], self.transThrs[i], \
                self.oriThrs[i]) for i in range(self.criterion_num)]

        self.maxDets = [100]
        self.useCats = 0



if __name__ == '__main__':
    import cv2
    parser = argparse.ArgumentParser(
        description='Evaluation self 3d car detection.')
    parser.add_argument('--test_dir', default='./test_eval_data/det3d_res/',
                        help='the dir of results')
    parser.add_argument('--gt_dir', default='./test_eval_data/det3d_gt/',
                        help='the dir of ground truth')
    parser.add_argument('--res_file', default='./test_eval_data/res.txt',
                        help='the dir of save results')
    parser.add_argument('--simType', default=None,
                        help='the type of evalution metric, default 3dpose')
    parser.add_argument('-light', '--light', dest='light', action='store_true', help='use light version')   
    args = parser.parse_args()
    det_3d_metric = Detect3DEval(args)
    det_3d_metric.load_car_models()
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


