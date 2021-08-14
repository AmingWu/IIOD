# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths


import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss

from model.utils.parser_func import parse_args, set_dataset_args

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data_t = torch.FloatTensor(1)
    im_info_t = torch.FloatTensor(1)
    num_boxes_t = torch.LongTensor(1)
    gt_boxes_t = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        im_data_t = im_data_t.cuda()
        im_info_t = im_info_t.cuda()
        num_boxes_t = num_boxes_t.cuda()
        gt_boxes_t = gt_boxes_t.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    im_data_t = Variable(im_data_t)
    im_info_t = Variable(im_info_t)
    num_boxes_t = Variable(num_boxes_t)
    gt_boxes_t = Variable(gt_boxes_t)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    # from model.faster_rcnn.vgg16_global_local import vgg16
    from model.faster_rcnn.resnet_global_local import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=args.lc,
                           gc=args.gc)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    paramtxt = open('update/RCNN_base1.txt', 'r')
    param = paramtxt.readlines()
    RCNN_base1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_base1.append(name)

    paramtxt = open('update/RCNN_base2.txt', 'r')
    param = paramtxt.readlines()
    RCNN_base2 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_base2.append(name)

    paramtxt = open('update/netD_pixel.txt', 'r')
    param = paramtxt.readlines()
    netD_pixel = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_pixel.append(name)

    paramtxt = open('update/netD_pixel1.txt', 'r')
    param = paramtxt.readlines()
    netD_pixel1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_pixel1.append(name)

    paramtxt = open('update/netD_base.txt', 'r')
    param = paramtxt.readlines()
    netD_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_base.append(name)

    paramtxt = open('update/RCNN_top_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_top_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_top_base.append(name)

    paramtxt = open('update/RCNN_cls_score_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_cls_score_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_cls_score_base.append(name)

    paramtxt = open('update/RCNN_bbox_pred_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_bbox_pred_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_bbox_pred_base.append(name)

    paramtxt = open('update/RCNN_rpn.txt', 'r')
    param = paramtxt.readlines()
    RCNN_rpn = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_rpn.append(name)

    paramtxt = open('update/di.txt', 'r')
    param = paramtxt.readlines()
    di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        di.append(name)

    paramtxt = open('update/ds.txt', 'r')
    param = paramtxt.readlines()
    ds = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        ds.append(name)

    paramtxt = open('update/di1.txt', 'r')
    param = paramtxt.readlines()
    di1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        di1.append(name)

    paramtxt = open('update/ds1.txt', 'r')
    param = paramtxt.readlines()
    ds1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        ds1.append(name)

    paramtxt = open('update/M.txt', 'r')
    param = paramtxt.readlines()
    M = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        M.append(name)

    paramtxt = open('update/M1.txt', 'r')
    param = paramtxt.readlines()
    M1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        M1.append(name)

    paramtxt = open('update/netD_ds.txt', 'r')
    param = paramtxt.readlines()
    netD_ds = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_ds.append(name)

    paramtxt = open('update/RCNN_bbox_pred_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_bbox_pred_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_bbox_pred_di.append(name)

    paramtxt = open('update/RCNN_cls_score_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_cls_score_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_cls_score_di.append(name)

    paramtxt = open('update/RCNN_top_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_top_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_top_di.append(name)

    paramtxt = open('update/recon.txt', 'r')
    param = paramtxt.readlines()
    Recon = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        Recon.append(name)

    di_p = []; ds_p = []; di1_p = []; ds1_p = []; M_p = []; M1_p = []; netD_base_p = []; netD_pixel_p = []; netD_pixel1_p = []; netD_ds_p = []; RCNN_base1_p = []; RCNN_base2_p = [];
    RCNN_bbox_pred_base_p = []; RCNN_bbox_pred_di_p = []; RCNN_cls_score_base_p = [];
    RCNN_cls_score_di_p = []; RCNN_rpn_p = []; RCNN_top_base_p = []; RCNN_top_di_p = []; Recon_p = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if key in RCNN_top_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_top_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_top_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if key in di:
            if value.requires_grad:
                if 'bias' in key:
                    di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in ds:
            if value.requires_grad:
                if 'bias' in key:
                    ds_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    ds_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in di1:
            if value.requires_grad:
                if 'bias' in key:
                    di1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    di1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in ds1:
            if value.requires_grad:
                if 'bias' in key:
                    ds1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    ds1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_pixel:
            if value.requires_grad:
                if 'bias' in key:
                    netD_pixel_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_pixel_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_pixel1:
            if value.requires_grad:
                if 'bias' in key:
                    netD_pixel1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_pixel1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in M:
            if value.requires_grad:
                if 'bias' in key:
                    M_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    M_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in M1:
            if value.requires_grad:
                if 'bias' in key:
                    M1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    M1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_base:
            if value.requires_grad:
                if 'bias' in key:
                    netD_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_ds:
            if value.requires_grad:
                if 'bias' in key:
                    netD_ds_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_ds_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_base1:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_base1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_base1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_base2:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_base2_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_base2_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_bbox_pred_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_bbox_pred_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_bbox_pred_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_bbox_pred_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_bbox_pred_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_bbox_pred_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_cls_score_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_cls_score_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_cls_score_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_cls_score_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_cls_score_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_cls_score_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_rpn:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_rpn_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_rpn_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_top_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_top_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_top_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in Recon:
            if value.requires_grad:
                if 'bias' in key:
                    Recon_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    Recon_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    opt_di = torch.optim.SGD(di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_ds = torch.optim.SGD(ds_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_M = torch.optim.SGD(M_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_pixel = torch.optim.SGD(netD_pixel_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_di1 = torch.optim.SGD(di1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_ds1 = torch.optim.SGD(ds1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_M1 = torch.optim.SGD(M1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_pixel1 = torch.optim.SGD(netD_pixel1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_base = torch.optim.SGD(netD_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_ds = torch.optim.SGD(netD_ds_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_base1 = torch.optim.SGD(RCNN_base1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_base2 = torch.optim.SGD(RCNN_base2_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_bbox_pred_base = torch.optim.SGD(RCNN_bbox_pred_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_bbox_pred_di = torch.optim.SGD(RCNN_bbox_pred_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_cls_score_base = torch.optim.SGD(RCNN_cls_score_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_cls_score_di = torch.optim.SGD(RCNN_cls_score_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_rpn = torch.optim.SGD(RCNN_rpn_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_top_base = torch.optim.SGD(RCNN_top_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_top_di = torch.optim.SGD(RCNN_top_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_Recon = torch.optim.SGD(Recon_p, momentum=cfg.TRAIN.MOMENTUM)

    optimizer = [opt_di, opt_ds, opt_M, opt_di1, opt_ds1, opt_M1, opt_netD_base, opt_netD_ds, opt_RCNN_base1, opt_RCNN_base2, opt_RCNN_bbox_pred_base, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_base, \
    opt_RCNN_cls_score_di, opt_RCNN_rpn, opt_RCNN_top_base, opt_RCNN_top_di, opt_Recon, opt_netD_pixel, opt_netD_pixel1]

    def reset_grad():
        opt_di.zero_grad()
        opt_ds.zero_grad()
        opt_M.zero_grad()
        opt_di1.zero_grad()
        opt_ds1.zero_grad()
        opt_M1.zero_grad()
        opt_netD_base.zero_grad()
        opt_netD_ds.zero_grad()
        opt_RCNN_base1.zero_grad()
        opt_RCNN_base2.zero_grad()
        opt_RCNN_bbox_pred_base.zero_grad()
        opt_RCNN_bbox_pred_di.zero_grad()
        opt_RCNN_cls_score_di.zero_grad()
        opt_RCNN_cls_score_base.zero_grad()
        opt_RCNN_rpn.zero_grad()
        opt_RCNN_top_base.zero_grad()
        opt_RCNN_top_di.zero_grad()
        opt_Recon.zero_grad()
        opt_netD_pixel.zero_grad()
        opt_netD_pixel1.zero_grad()

    def group_step(step_list):
        for i in range(len(step_list)):
            step_list[i].step()
        reset_grad()

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
    count_iter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            lr_decay = args.lr_decay_gamma
            for m in range(len(optimizer)):
               adjust_learning_rate(optimizer[m], lr_decay)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            eta = 1.0
            count_iter += 1
            #put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes_t.data.resize_(1, 1, 5).fill_(1.0)
            num_boxes_t.data.resize_(1).zero_()

            #First Step
            fasterRCNN.zero_grad()
            reset_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d, out_ds = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=1, eta=eta)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + (RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean() \
                   + RCNN_loss_cls[1].mean() + RCNN_loss_bbox[1].mean()) * 0.5
            domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
            dloss_s = 0.5 * FL(out_d, domain_s)
            domain_s_d = Variable(torch.zeros(out_ds.size(0)).long().cuda())
            dloss_s_d = 0.5 * FL(out_ds, domain_s_d)
            dloss_s_p0 = 0.5 * torch.mean(out_d_pixel[0] ** 2)
            dloss_s_p1 = 0.5 * torch.mean(out_d_pixel[1] ** 2)

            out_d_pixel, out_d, out_d_ds = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, phase=1, target=True, eta=eta)
            domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_d, domain_t)
            domain_t_d = Variable(torch.ones(out_d_ds.size(0)).long().cuda())
            dloss_t_d = 0.5 * FL(out_d_ds, domain_t_d)
            dloss_t_p0 = 0.5 * torch.mean((1 - out_d_pixel[0]) ** 2)
            dloss_t_p1 = 0.5 * torch.mean((1 - out_d_pixel[1]) ** 2)

            loss += (dloss_s + dloss_t + dloss_s_d + dloss_t_d) + (dloss_s_p0 + dloss_t_p0 + dloss_s_p1 + dloss_t_p1) * 0.5

            loss.backward()
            group_step([opt_RCNN_base1, opt_RCNN_base2, opt_netD_pixel, opt_netD_pixel1, opt_netD_base, opt_RCNN_bbox_pred_base, opt_RCNN_cls_score_base, opt_RCNN_top_base, opt_RCNN_rpn, \
                opt_di, opt_ds, opt_di1, opt_ds1, opt_netD_ds, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_di, opt_RCNN_top_di])

            #Second Step
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d, out_d_pixel, MI_s, adj_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=2, eta=eta)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + (RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean())
            domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
            dloss_s = 0.5 * FL(out_d, domain_s)
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)
            loss_temp += (rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean()).item()

            out_d, out_d_pixel, MI_t, adj_t = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, phase=2, target=True, eta=eta)
            domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_d, domain_t)
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)

            MI_s = (MI_s[0] + MI_s[1]) * 0.5
            MI_t = (MI_t[0] + MI_t[1]) * 0.5
            loss += (dloss_s + dloss_t + MI_s + MI_t + adj_s + adj_t) + (dloss_s_p + dloss_t_p)

            loss.backward()
            group_step([opt_di, opt_ds, opt_di1, opt_ds1, opt_netD_ds, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_di, opt_RCNN_top_di, opt_M, opt_M1])

            #Third Step
            fasterRCNN.zero_grad()
            recon_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=3, eta=eta)
            loss = 0.0

            recon_t = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, phase=3, target=True, eta=eta)

            loss += (recon_s + recon_t) * 0.5 * 0.01

            loss.backward()
            group_step([opt_di, opt_ds, opt_Recon])

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls[0].item()
                    loss_rcnn_box = RCNN_loss_bbox[0].item()
                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    MI_s = recon_s.item()
                    MI_t = recon_t.item()

                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss s: %.4f dloss t: %.4f dloss s pixel: %.4f dloss t pixel: %.4f recon_s: %.4f recon_t: %.4f adj_s: %.4f adj_t: %.4f eta: %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s, dloss_t, dloss_s_p, dloss_t_p, recon_s, recon_t, adj_s, adj_t, args.eta))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()
        save_name = os.path.join(output_dir,
                                 'globallocal_target_{}_eta_{}_local_context_{}_global_context_{}_gamma_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.lc, args.gc, args.gamma,
                                     args.session, epoch,
                                     step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
