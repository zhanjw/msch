import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

# dataset can be 'MIRFlickr' or 'NUSWIDE'
DATASET = 'MIRFlickr'

if DATASET == 'MIRFlickr':
    LABEL_DIR = './data/Flickr/mirflickr25k-lall.mat'
    TXT_DIR = './data/Flickr/mirflickr25k-yall.mat'
    IMG_DIR = './data/Flickr/mirflickr25k-iall.mat'

    GAMMA1 = 0.7
    GAMMA2 = 0.4
    GAMMA3 = 0.3
    NUM_EPOCH = 160
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 40

if DATASET == 'NUSWIDE':
    LABEL_DIR = './data/NUS-WIDE-TC21/nus-wide-tc21-lall.mat'
    TXT_DIR = './data/NUS-WIDE-TC21/nus-wide-tc21-yall.mat'
    IMG_DIR = './data/NUS-WIDE-TC21/nus-wide-tc21-iall.mat'

    GAMMA1 = 0.8
    GAMMA2 = 0.4
    GAMMA3 = 0.3
    NUM_EPOCH = 160
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 40

ALPHA = 1.1
BETA = 0.9
LAMBDA = 0.4
EPSILON = 5.0

BATCH_SIZE = 128
CODE_LEN = 32

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 8
EPOCH_INTERVAL = 1

MODEL_DIR = './checkpoint'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('ALPHA = %.4f' % ALPHA)
logger.info('BETA = %.4f' % BETA)
logger.info('LAMBDA = %.4f' % LAMBDA)
logger.info('EPSILON = %.4f' % EPSILON)
logger.info('GAMMA1 = %.4f' % GAMMA1)
logger.info('GAMMA2 = %.4f' % GAMMA2)
logger.info('GAMMA3 = %.4f' % GAMMA3)
# logger.info('LAMBDA1 = %.4f' % LAMBDA1)
# logger.info('LAMBDA2 = %.4f' % LAMBDA2)
logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('CODE_LEN = %d' % CODE_LEN)
# logger.info('MU = %.4f' % MU)
# logger.info('ETA = %.4f' % ETA)
logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')
