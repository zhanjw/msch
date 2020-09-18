import datasets_fir_32 as datasets
import numpy as np
import os
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ImgNet
from models import TxtNet
from msloss import *
from torch.autograd import Variable
from utils import *

import settings_fir_32 as settings
#import setproctitle
#setproctitle.setproctitle("python")


class Session:
    def __init__(self):
        self.logger = settings.logger
        # torch.cuda.set_device(settings.GPU_ID)

        if settings.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True,
                                                       transform=datasets.mir_test_transform)

        if settings.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        self.train_l = torch.from_numpy(self.train_dataset.train_labels).cuda()
        self.test_l = torch.from_numpy(self.test_dataset.train_labels).cuda()
        self.database_l = torch.from_numpy(self.database_dataset.train_labels).cuda()

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=settings.BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=settings.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=settings.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=settings.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=settings.BATCH_SIZE,
                                                           shuffle=False,
                                                           num_workers=settings.NUM_WORKERS)

        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN).cuda()
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN).cuda().eval()
        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len).cuda()

        self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=np.power(0.1, 4.0))
        self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=np.power(0.1, 3.5))

        self.logger.info("train length: %d" % len(self.train_dataset))
        self.logger.info("query length: %d" % len(self.test_dataset))
        self.logger.info("database length: %d" % len(self.database_dataset))

    def train(self, epoch):
        self.CodeNet_I.train()
        self.CodeNet_T.train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, F_T, labels, ind) in enumerate(self.train_loader):
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            F_I, hid_I, code_I = self.CodeNet_I(img)
            _, hid_T, code_T = self.CodeNet_T(F_T)

            F_I = F.normalize(F_I)
            S_I = F_I.mm(F_I.t())
            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())

            S_prime = settings.GAMMA1 * S_I + (1 - settings.GAMMA1) * S_T
            S_dprime = (1 - settings.GAMMA2) * S_prime + settings.GAMMA2 * S_prime.mm(S_prime) / settings.BATCH_SIZE
            S_dprime = S_dprime * 1.5

            # similar matrix size: (batch_size, num_train)
            S_pos = calc_neighbor(labels, labels).float()  # S: (batch_size, batch_size)

            S_rev = torch.max(S_dprime) - S_dprime + torch.min(S_dprime)
            S_neg = torch.where(S_pos < 0.5, (1 - S_pos) * settings.GAMMA3 + S_rev * (1 - settings.GAMMA3), 1 - S_pos)
            S_pos = torch.where(S_pos > 0.5, S_pos * settings.GAMMA3 + S_dprime * (1 - settings.GAMMA3), S_pos)

            loss = mslossx(S_pos, S_neg, code_I, code_T, scale_pos=settings.ALPHA, scale_neg=settings.BETA,
                           thresh=settings.LAMBDA, ms_mining=True, ms_margin=settings.EPSILON)
            loss.backward()

            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss.item()))

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                          self.CodeNet_T, self.database_dataset, self.test_dataset)

        p_i2t, r_i2t = pr_curve(qu_BI, re_BT, qu_L, re_L)
        p_t2i, r_t2i = pr_curve(qu_BT, re_BI, qu_L, re_L)

        K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        pk_i2t = p_topK(qu_BI, re_BT, qu_L, re_L, K)
        pk_t2i = p_topK(qu_BT, re_BI, qu_L, re_L, K)

        path = 'checkpoint/'
        np.save(os.path.join(path, 'msch_fir_32_P_i2t.npy'), p_i2t.numpy())
        np.save(os.path.join(path, 'msch_fir_32_R_i2t.npy'), r_i2t.numpy())
        np.save(os.path.join(path, 'msch_fir_32_P_t2i.npy'), p_t2i.numpy())
        np.save(os.path.join(path, 'msch_fir_32_R_t2i.npy'), r_t2i.numpy())
        np.save(os.path.join(path, 'msch_fir_32_P_at_K_i2t.npy'), pk_i2t.numpy())
        np.save(os.path.join(path, 'msch_fir_32_P_at_K_t2i.npy'), pk_t2i.numpy())

        mapi2t = calc_map_k(qu_BI, re_BT, qu_L, re_L)
        mapt2i = calc_map_k(qu_BT, re_BI, qu_L, re_L)

        self.logger.info('MAP of Image to Text: %.7f, MAP of Text to Image: %.7f' % (mapi2t, mapt2i))
        self.logger.info('--------------------------------------------------------------------')

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).cuda()
    return Sim


def main():
    sess = Session()
    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()
    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # save the model
            if epoch + 1 == settings.NUM_EPOCH:
                sess.save_checkpoints(step=epoch + 1, file_name='mjch.pth')


if __name__ == '__main__':
    main()
