from detectron2.config import CfgNode as CN

def add_default_config(cfg):
    _C = cfg
    _C.MODEL.DEVICE = 'cuda'

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/lvis/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT = 200
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = 'GN'
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 
    _C.INPUT.NOT_CLAMP_BOX = False
    
    _C.DEBUG = False
    _C.SAVE_DEBUG = False
    _C.SAVE_PTH = False
    _C.VIS_THRESH = 0.3
    _C.DEBUG_SHOW_NAME = False

    _C.DATASETS.RESIZE = (1500,1873)