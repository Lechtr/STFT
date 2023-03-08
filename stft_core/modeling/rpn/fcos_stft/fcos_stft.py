import os
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn

from stft_core.layers import Scale
from stft_core.layers import DFConv2d
from stft_core.layers import DeformConv
from .inference import make_fcos_stft_postprocessor
from .loss import make_fcos_stft_loss_evaluator


class SFTBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(SFTBranch, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        offset_input_channel = 4
        self.conv_offset = nn.Conv2d(offset_input_channel, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, offset_weight_std=0.01):
        torch.nn.init.normal_(self.conv_offset.weight, mean=0, std=offset_weight_std)
        if hasattr(self.conv_offset, 'bias') and self.conv_offset.bias is not None:
            torch.nn.init.constant_(self.conv_offset.bias, 0)
        torch.nn.init.normal_(self.conv_adaption.weight, mean=0, std=0.01)
        if hasattr(self.conv_adaption, 'bias') and self.conv_adaption.bias is not None:
            torch.nn.init.constant_(self.conv_adaption.bias, 0)

    def forward(self, feature, pred_shape):
        pred_shape = pred_shape.permute(0, 2, 1).reshape(feature.shape[0], -1, feature.shape[2], feature.shape[3])
        offset = self.conv_offset(pred_shape.detach().contiguous())
        feature = self.relu(self.conv_adaption(feature, offset))
        return feature


class STFTFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(STFTFCOSHead, self).__init__()
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.add_module("dcn_cls_subnet", SFTBranch(in_channels, in_channels))
        self.add_module("dcn_bbox_subnet", SFTBranch(in_channels, in_channels))

        self.dcn_cls_score = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.dcn_bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                if isinstance(l, nn.GroupNorm):
                    torch.nn.init.constant_(l.weight, 1)
                    torch.nn.init.constant_(l.bias, 0)

        self.dcn_cls_subnet.init_weights(offset_weight_std=cfg.MODEL.STFT.OFFSET_WEIGHT_STD)
        self.dcn_bbox_subnet.init_weights(offset_weight_std=cfg.MODEL.STFT.OFFSET_WEIGHT_STD)

        for modules in [self.dcn_cls_score, self.dcn_bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.dcn_cls_score.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])


    def forward(self, x, shifts):
        logits = []
        bbox_reg = []
        centerness = []
        stft_logits = []
        stft_bbox_reg = []
        pre_bbox = []

        shifts = [
            torch.cat([shi.unsqueeze(0) for shi in shift], dim=0) 
            for shift in list(zip(*shifts))
        ]

        for l, (feature, shifts_i) in enumerate(zip(x, shifts)):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            # only record target frame
            cls_logits = self.cls_logits(cls_tower)
            logits.append(cls_logits[0].unsqueeze(0))

            if self.centerness_on_reg:
                centerness_logits = self.centerness(box_tower)
            else:
                centerness_logits = self.centerness(cls_tower)
            centerness.append(centerness_logits[0].unsqueeze(0))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred) * self.fpn_strides[l]
            else:
                bbox_pred = torch.exp(bbox_pred) * self.fpn_strides[l]
            bbox_reg.append(bbox_pred[0].unsqueeze(0))

            ###### STFT
            N, C, H, W = feature.shape
            pre_off = bbox_pred.clone().detach()
            with torch.no_grad():
                pre_off = pre_off.permute(0, 2, 3, 1).reshape(N, -1, 4) #l,t,r,b
                pre_boxes = self.compute_bbox(shifts_i, pre_off) #x1,y1,x2,y2
                pre_bbox.append(pre_boxes[0].unsqueeze(0))

                #align on feature map scale
                align_boxes, wh = self.compute_border(pre_boxes, l, H, W) #x1,y1,x2,y2,  w,h
                align_ltrb_off = self.compute_ltrb_off(align_boxes, shifts_i, l, H, W) #l,t,r,b

            align_boxes[1:] = align_boxes[0] - align_boxes[1:]
            wh[1:] = wh[0] - wh[1:]
            align_ltrb_off[1:] = align_ltrb_off[0] - align_ltrb_off[1:]

            # STFT -- classification
            stft_cls_feats = self.dcn_cls_subnet(cls_tower, align_ltrb_off)
            # channel-aware
            target_stft_cls_feats = stft_cls_feats[0].unsqueeze(0).permute(1, 0, 2, 3)
            target_stft_cls_feats = target_stft_cls_feats.reshape(stft_cls_feats.shape[1], 1, -1)
            support_stft_cls_feats = stft_cls_feats[1:].permute(1, 0, 2, 3)
            support_stft_cls_feats = support_stft_cls_feats.reshape(stft_cls_feats.shape[1], -1, stft_cls_feats.shape[2]*stft_cls_feats.shape[3])
            sim_stft_cls = torch.bmm(target_stft_cls_feats, support_stft_cls_feats.transpose(1, 2)) 
            sim_stft_cls = (1.0 / math.sqrt(float(support_stft_cls_feats.shape[2]))) * sim_stft_cls
            sim_stft_cls = F.softmax(sim_stft_cls, dim=2)
            att_stft_cls = torch.bmm(sim_stft_cls, support_stft_cls_feats)

            target_stft_cls_feats = target_stft_cls_feats + att_stft_cls
            target_stft_cls_feats = target_stft_cls_feats.reshape(stft_cls_feats.shape[1], stft_cls_feats.shape[2], stft_cls_feats.shape[3])
            target_stft_cls_feats = target_stft_cls_feats.unsqueeze(0)
            
            stft_cls_logits = self.dcn_cls_score(target_stft_cls_feats)
            stft_logits.append(stft_cls_logits)

            # STFT -- regression
            stft_reg_feats = self.dcn_bbox_subnet(box_tower, align_ltrb_off)
            # channel-aware
            target_stft_reg_feats = stft_reg_feats[0].unsqueeze(0).permute(1, 0, 2, 3)
            target_stft_reg_feats = target_stft_reg_feats.reshape(stft_reg_feats.shape[1], 1, -1)
            support_stft_reg_feats = stft_reg_feats[1:].permute(1, 0, 2, 3)
            support_stft_reg_feats = support_stft_reg_feats.reshape(stft_reg_feats.shape[1], -1, stft_reg_feats.shape[2]*stft_reg_feats.shape[3])
            sim_stft_reg = torch.bmm(target_stft_reg_feats, support_stft_reg_feats.transpose(1, 2)) 
            sim_stft_reg = (1.0 / math.sqrt(float(support_stft_reg_feats.shape[2]))) * sim_stft_reg
            sim_stft_reg = F.softmax(sim_stft_reg, dim=2)
            att_stft_reg = torch.bmm(sim_stft_reg, support_stft_reg_feats)

            target_stft_reg_feats = target_stft_reg_feats + att_stft_reg
            target_stft_reg_feats = target_stft_reg_feats.reshape(stft_reg_feats.shape[1], stft_reg_feats.shape[2], stft_reg_feats.shape[3])
            target_stft_reg_feats = target_stft_reg_feats.unsqueeze(0)

            stft_bbox_reg_pred = self.dcn_bbox_pred(target_stft_reg_feats)
            stft_bbox_reg.append(stft_bbox_reg_pred)

        if self.training:
            pre_bbox = torch.cat(pre_bbox, dim=1)
        return logits, bbox_reg, centerness, stft_logits, stft_bbox_reg, pre_bbox


    def compute_bbox(self, location, pred_offset):
        detections = torch.stack([
            location[:, :, 0] - pred_offset[:, :, 0],
            location[:, :, 1] - pred_offset[:, :, 1],
            location[:, :, 0] + pred_offset[:, :, 2],
            location[:, :, 1] + pred_offset[:, :, 3]], dim=2)

        return detections

    def compute_border(self, _boxes, fm_i, height, width):
        boxes = _boxes / self.fpn_strides[fm_i]
        boxes[:, :, 0].clamp_(min=0, max=width - 1)
        boxes[:, :, 1].clamp_(min=0, max=height - 1)
        boxes[:, :, 2].clamp_(min=0, max=width - 1)
        boxes[:, :, 3].clamp_(min=0, max=height - 1)

        wh = (boxes[:, :, 2:] - boxes[:, :, :2]).contiguous()
        return boxes, wh

    def compute_ltrb_off(self, align_boxes, location, fm_i, height, width):
        align_loc = location / self.fpn_strides[fm_i]
        align_loc[:, :, 0].clamp_(min=0, max=width - 1)
        align_loc[:, :, 1].clamp_(min=0, max=height - 1)

        align_ltrb = torch.stack([
            align_boxes[:, :, 0] - align_loc[:, :, 0],
            align_boxes[:, :, 1] - align_loc[:, :, 1],
            align_boxes[:, :, 2] - align_loc[:, :, 0],
            align_boxes[:, :, 3] - align_loc[:, :, 1]], dim=2)
        return align_ltrb


class STFTFCOSModule(torch.nn.Module):
    """
    Module for STFTFCOS computation. Takes feature maps from the backbone and
    STFTFCOS outputs and losses. Only Test on FPN now.
    """
    def __init__(self, cfg, in_channels):
        super(STFTFCOSModule, self).__init__()
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.head = STFTFCOSHead(cfg, in_channels)
        self.box_selector_test = make_fcos_stft_postprocessor(cfg)
        self.loss_evaluator = make_fcos_stft_loss_evaluator(cfg)

    def forward(self, images, features, targets=None):
        locations = self.compute_locations(features)
        shifts = [
            copy.deepcopy(locations)
            for _ in range(images.shape[0])
        ]

        box_cls, box_regression, centerness, stft_box_cls, stft_box_reg, stft_based_box = self.head(features, shifts)

        if self.training:
            return self._forward_train(
                [shifts[0]], box_cls, box_regression, centerness, targets,
                stft_based_box, stft_box_cls, stft_box_reg
            )
        else:
            return self._forward_test(
                [shifts[0]], box_cls, centerness, 
                stft_box_cls, stft_box_reg, stft_based_box, [(images.shape[2], images.shape[3])])

    def _forward_train(self, shifts, box_cls, box_regression, centerness, targets, stft_based_box, stft_box_cls, stft_box_reg):
        loss_cls, loss_box_reg, loss_centerness, loss_stft_cls, loss_stft_reg = self.loss_evaluator(
            shifts, box_cls, box_regression, centerness, targets, stft_based_box, stft_box_cls, stft_box_reg
        )

        # shifts: [[tensor([[   4.,    4.],
        #         [  12.,    4.],
        #         [  20.,    4.],
        #         ...,
        #         [1052.,  796.],
        #         [1060.,  796.],
        #         [1068.,  796.]], device='cuda:0'), tensor([[   8.,    8.],
        #         [  24.,    8.],
        #         [  40.,    8.],
        #         ...,
        #         [1032.,  792.],
        #         [1048.,  792.],
        #         [1064.,  792.]], device='cuda:0'), tensor([[  16.,   16.],
        #         [  48.,   16.],
        #         [  80.,   16.],
        #         ...,
        #         [1008.,  784.],
        #         [1040.,  784.],
        #         [1072.,  784.]], device='cuda:0'), tensor([[  32.,   32.],
        #         [  96.,   32.],
        #         [ 160.,   32.],
        #         [ 224.,   32.],
        #         [ 288.,   32.],
        #         [ 352.,   32.],
        #         [ 416.,   32.],
        #         [ 480.,   32.],
        #         [ 544.,   32.],
        #         [ 608.,   32.],
        #         [ 672.,   32.],
        #         [ 736.,   32.],
        #         [ 800.,   32.],
        #         [ 864.,   32.],
        #         [ 928.,   32.],
        #         [ 992.,   32.],
        #         [1056.,   32.],
        #         [  32.,   96.],
        #         [  96.,   96.],
        #         [ 160.,   96.],
        #         [ 224.,   96.],
        #         [ 288.,   96.],
        #         [ 352.,   96.],
        #         [ 416.,   96.],
        #         [ 480.,   96.],
        #         [ 544.,   96.],
        #         [ 608.,   96.],
        #         [ 672.,   96.],
        #         [ 736.,   96.],
        #         [ 800.,   96.],
        #         [ 864.,   96.],
        #         [ 928.,   96.],
        #         [ 992.,   96.],
        #         [1056.,   96.],
        #         [  32.,  160.],
        #         [  96.,  160.],
        #         [ 160.,  160.],
        #         [ 224.,  160.],
        #         [ 288.,  160.],
        #         [ 352.,  160.],
        #         [ 416.,  160.],
        #         [ 480.,  160.],
        #         [ 544.,  160.],
        #         [ 608.,  160.],
        #         [ 672.,  160.],
        #         [ 736.,  160.],
        #         [ 800.,  160.],
        #         [ 864.,  160.],
        #         [ 928.,  160.],
        #         [ 992.,  160.],
        #         [1056.,  160.],
        #         [  32.,  224.],
        #         [  96.,  224.],
        #         [ 160.,  224.],
        #         [ 224.,  224.],
        #         [ 288.,  224.],
        #         [ 352.,  224.],
        #         [ 416.,  224.],
        #         [ 480.,  224.],
        #         [ 544.,  224.],
        #         [ 608.,  224.],
        #         [ 672.,  224.],
        #         [ 736.,  224.],
        #         [ 800.,  224.],
        #         [ 864.,  224.],
        #         [ 928.,  224.],
        #         [ 992.,  224.],
        #         [1056.,  224.],
        #         [  32.,  288.],
        #         [  96.,  288.],
        #         [ 160.,  288.],
        #         [ 224.,  288.],
        #         [ 288.,  288.],
        #         [ 352.,  288.],
        #         [ 416.,  288.],
        #         [ 480.,  288.],
        #         [ 544.,  288.],
        #         [ 608.,  288.],
        #         [ 672.,  288.],
        #         [ 736.,  288.],
        #         [ 800.,  288.],
        #         [ 864.,  288.],
        #         [ 928.,  288.],
        #         [ 992.,  288.],
        #         [1056.,  288.],
        #         [  32.,  352.],
        #         [  96.,  352.],
        #         [ 160.,  352.],
        #         [ 224.,  352.],
        #         [ 288.,  352.],
        #         [ 352.,  352.],
        #         [ 416.,  352.],
        #         [ 480.,  352.],
        #         [ 544.,  352.],
        #         [ 608.,  352.],
        #         [ 672.,  352.],
        #         [ 736.,  352.],
        #         [ 800.,  352.],
        #         [ 864.,  352.],
        #         [ 928.,  352.],
        #         [ 992.,  352.],
        #         [1056.,  352.],
        #         [  32.,  416.],
        #         [  96.,  416.],
        #         [ 160.,  416.],
        #         [ 224.,  416.],
        #         [ 288.,  416.],
        #         [ 352.,  416.],
        #         [ 416.,  416.],
        #         [ 480.,  416.],
        #         [ 544.,  416.],
        #         [ 608.,  416.],
        #         [ 672.,  416.],
        #         [ 736.,  416.],
        #         [ 800.,  416.],
        #         [ 864.,  416.],
        #         [ 928.,  416.],
        #         [ 992.,  416.],
        #         [1056.,  416.],
        #         [  32.,  480.],
        #         [  96.,  480.],
        #         [ 160.,  480.],
        #         [ 224.,  480.],
        #         [ 288.,  480.],
        #         [ 352.,  480.],
        #         [ 416.,  480.],
        #         [ 480.,  480.],
        #         [ 544.,  480.],
        #         [ 608.,  480.],
        #         [ 672.,  480.],
        #         [ 736.,  480.],
        #         [ 800.,  480.],
        #         [ 864.,  480.],
        #         [ 928.,  480.],
        #         [ 992.,  480.],
        #         [1056.,  480.],
        #         [  32.,  544.],
        #         [  96.,  544.],
        #         [ 160.,  544.],
        #         [ 224.,  544.],
        #         [ 288.,  544.],
        #         [ 352.,  544.],
        #         [ 416.,  544.],
        #         [ 480.,  544.],
        #         [ 544.,  544.],
        #         [ 608.,  544.],
        #         [ 672.,  544.],
        #         [ 736.,  544.],
        #         [ 800.,  544.],
        #         [ 864.,  544.],
        #         [ 928.,  544.],
        #         [ 992.,  544.],
        #         [1056.,  544.],
        #         [  32.,  608.],
        #         [  96.,  608.],
        #         [ 160.,  608.],
        #         [ 224.,  608.],
        #         [ 288.,  608.],
        #         [ 352.,  608.],
        #         [ 416.,  608.],
        #         [ 480.,  608.],
        #         [ 544.,  608.],
        #         [ 608.,  608.],
        #         [ 672.,  608.],
        #         [ 736.,  608.],
        #         [ 800.,  608.],
        #         [ 864.,  608.],
        #         [ 928.,  608.],
        #         [ 992.,  608.],
        #         [1056.,  608.],
        #         [  32.,  672.],
        #         [  96.,  672.],
        #         [ 160.,  672.],
        #         [ 224.,  672.],
        #         [ 288.,  672.],
        #         [ 352.,  672.],
        #         [ 416.,  672.],
        #         [ 480.,  672.],
        #         [ 544.,  672.],
        #         [ 608.,  672.],
        #         [ 672.,  672.],
        #         [ 736.,  672.],
        #         [ 800.,  672.],
        #         [ 864.,  672.],
        #         [ 928.,  672.],
        #         [ 992.,  672.],
        #         [1056.,  672.],
        #         [  32.,  736.],
        #         [  96.,  736.],
        #         [ 160.,  736.],
        #         [ 224.,  736.],
        #         [ 288.,  736.],
        #         [ 352.,  736.],
        #         [ 416.,  736.],
        #         [ 480.,  736.],
        #         [ 544.,  736.],
        #         [ 608.,  736.],
        #         [ 672.,  736.],
        #         [ 736.,  736.],
        #         [ 800.,  736.],
        #         [ 864.,  736.],
        #         [ 928.,  736.],
        #         [ 992.,  736.],
        #         [1056.,  736.],
        #         [  32.,  800.],
        #         [  96.,  800.],
        #         [ 160.,  800.],
        #         [ 224.,  800.],
        #         [ 288.,  800.],
        #         [ 352.,  800.],
        #         [ 416.,  800.],
        #         [ 480.,  800.],
        #         [ 544.,  800.],
        #         [ 608.,  800.],
        #         [ 672.,  800.],
        #         [ 736.,  800.],
        #         [ 800.,  800.],
        #         [ 864.,  800.],
        #         [ 928.,  800.],
        #         [ 992.,  800.],
        #         [1056.,  800.]], device='cuda:0'), tensor([[  64.,   64.],
        #         [ 192.,   64.],
        #         [ 320.,   64.],
        #         [ 448.,   64.],
        #         [ 576.,   64.],
        #         [ 704.,   64.],
        #         [ 832.,   64.],
        #         [ 960.,   64.],
        #         [1088.,   64.],
        #         [  64.,  192.],
        #         [ 192.,  192.],
        #         [ 320.,  192.],
        #         [ 448.,  192.],
        #         [ 576.,  192.],
        #         [ 704.,  192.],
        #         [ 832.,  192.],
        #         [ 960.,  192.],
        #         [1088.,  192.],
        #         [  64.,  320.],
        #         [ 192.,  320.],
        #         [ 320.,  320.],
        #         [ 448.,  320.],
        #         [ 576.,  320.],
        #         [ 704.,  320.],
        #         [ 832.,  320.],
        #         [ 960.,  320.],
        #         [1088.,  320.],
        #         [  64.,  448.],
        #         [ 192.,  448.],
        #         [ 320.,  448.],
        #         [ 448.,  448.],
        #         [ 576.,  448.],
        #         [ 704.,  448.],
        #         [ 832.,  448.],
        #         [ 960.,  448.],
        #         [1088.,  448.],
        #         [  64.,  576.],
        #         [ 192.,  576.],
        #         [ 320.,  576.],
        #         [ 448.,  576.],
        #         [ 576.,  576.],
        #         [ 704.,  576.],
        #         [ 832.,  576.],
        #         [ 960.,  576.],
        #         [1088.,  576.],
        #         [  64.,  704.],
        #         [ 192.,  704.],
        #         [ 320.,  704.],
        #         [ 448.,  704.],
        #         [ 576.,  704.],
        #         [ 704.,  704.],
        #         [ 832.,  704.],
        #         [ 960.,  704.],
        #         [1088.,  704.],
        #         [  64.,  832.],
        #         [ 192.,  832.],
        #         [ 320.,  832.],
        #         [ 448.,  832.],
        #         [ 576.,  832.],
        #         [ 704.,  832.],
        #         [ 832.,  832.],
        #         [ 960.,  832.],
        #         [1088.,  832.]], device='cuda:0')]]

        # box_cls: [tensor([[[[-5.0307, -5.5677, -5.5694,  ..., -5.6206, -5.5880, -5.1044],
        #           [-5.1709, -5.5930, -5.5798,  ..., -5.6683, -5.6262, -4.9776],
        #           [-5.1651, -5.5687, -5.5475,  ..., -5.6587, -5.6200, -4.9662],
        #           ...,
        #           [-5.1914, -5.6284, -5.6289,  ..., -5.6974, -5.6453, -4.9780],
        #           [-5.1961, -5.6251, -5.6287,  ..., -5.6643, -5.6255, -4.9666],
        #           [-4.9159, -4.9156, -4.9128,  ..., -4.9245, -4.9173, -4.5261]],
        #
        #          [[-5.3260, -5.8105, -5.8199,  ..., -5.8900, -5.8893, -5.5969],
        #           [-5.7289, -5.9460, -5.9546,  ..., -6.0679, -6.0612, -5.7385],
        #           [-5.7264, -5.9336, -5.9381,  ..., -6.0796, -6.0601, -5.7454],
        #           ...,
        #           [-5.7653, -6.0111, -6.0426,  ..., -6.1464, -6.0967, -5.7547],
        #           [-5.7645, -6.0008, -6.0260,  ..., -6.0855, -6.0543, -5.7278],
        #           [-5.4461, -5.5152, -5.5235,  ..., -5.5482, -5.5295, -5.2855]]]],
        #        device='cuda:0', grad_fn=<UnsqueezeBackward0>), tensor([[[[-5.1419, -5.7950, -5.7786,  ..., -5.7752, -5.7745, -5.2300],
        #           [-5.3131, -5.8440, -5.8060,  ..., -5.8017, -5.8167, -5.0811],
        #           [-5.2948, -5.8001, -5.7424,  ..., -5.7529, -5.7927, -5.0650],
        #           ...,
        #           [-5.2917, -5.7942, -5.7312,  ..., -5.8287, -5.8322, -5.0814],
        #           [-5.3141, -5.8266, -5.7827,  ..., -5.8454, -5.8476, -5.0870],
        #           [-5.0002, -5.0099, -4.9823,  ..., -5.0151, -5.0287, -4.5656]],
        #
        #          [[-5.5077, -6.0885, -6.0890,  ..., -6.0809, -6.1141, -5.7853],
        #           [-6.0226, -6.3095, -6.3005,  ..., -6.2970, -6.3438, -5.9818],
        #           [-6.0038, -6.2725, -6.2502,  ..., -6.2585, -6.3176, -5.9787],
        #           ...,
        #           [-6.0030, -6.2795, -6.2597,  ..., -6.3458, -6.3705, -6.0028],
        #           [-6.0147, -6.2977, -6.2858,  ..., -6.3411, -6.3658, -5.9979],
        #           [-5.6420, -5.7359, -5.7248,  ..., -5.7571, -5.7648, -5.4701]]]],
        #        device='cuda:0', grad_fn=<UnsqueezeBackward0>), tensor([[[[-5.0952, -5.6118, -5.4104,  ..., -5.6087, -5.7258, -5.1712],
        #           [-5.0871, -5.3671, -5.0581,  ..., -5.4213, -5.6299, -4.9590],
        #           [-4.9622, -5.1912, -4.8505,  ..., -5.2767, -5.5468, -4.9172],
        #            ...
        #           [-6.0919, -6.2492, -6.0714,  ..., -6.2377, -6.4568, -6.1751],
        #           [-6.1841, -6.4033, -6.2763,  ..., -6.3990, -6.5480, -6.1969],
        #           [-5.7812, -5.8720, -5.8029,  ..., -5.8802, -5.9447, -5.6327]]]],
        #        device='cuda:0', grad_fn=<UnsqueezeBackward0>), tensor([[[[-4.8902, -5.2446, -4.8223, -4.4651, -4.1383, -3.9119, -3.8861,
        #            -3.9975, -4.1261, -4.1641, -4.2151, -4.2983, -4.4223, -4.6027,
        #            -4.8446, -5.2013, -4.7310],
        #            ...
        #           [-5.6012, -5.3894, -5.1991, -5.0749, -4.9968, -4.9490, -4.9970,
        #            -5.0775, -5.1335, -5.1610, -5.1423, -5.1142, -5.1309, -5.1925,
        #            -5.3141, -5.5480, -5.5302]]]], device='cuda:0',
        #        grad_fn=<UnsqueezeBackward0>), tensor([[[[-4.3546, -4.6025, -4.2653, -4.2518, -4.2823, -4.3161, -4.4909,
        #            -4.8548, -4.4109],
        #            ...
        #            -4.4507, -5.2320],
        #           [-5.1199, -4.5866, -4.3657, -4.2774, -4.2079, -4.1995, -4.3637,
        #            -4.7716, -5.1998]]]], device='cuda:0', grad_fn=<UnsqueezeBackward0>)]

        # box_regression
        # [tensor([[[[19.8454, 38.3197, 38.1193,  ..., 38.3437, 39.2014, 31.0694],
        #           [29.1303, 54.1570, 53.9444,  ..., 54.5293, 56.2565, 43.4438],
        #           [28.3872, 52.6910, 52.3415,  ..., 53.0500, 55.2676, 42.7488],
        #           ...,
        #           [27.6996, 50.4150, 49.2357,  ..., 47.2541, 50.5765, 40.6334],
        #           [28.5170, 52.0093, 51.0373,  ..., 49.8113, 52.7084, 41.8591],
        #           [21.3098, 34.8001, 34.1912,  ..., 34.1143, 35.8991, 28.0343]],

        # centerness
        # [tensor([[[[ 0.7705,  0.6291,  0.6728,  ...,  0.6480,  0.6123,  0.4923],
        #           [ 0.4672,  0.0052,  0.0840,  ...,  0.0560, -0.0159, -0.0374],
        #           [ 0.4955,  0.0948,  0.2008,  ...,  0.1829,  0.0962,  0.0323],
        #           ...,
        #           [ 0.5151,  0.1932,  0.3278,  ...,  0.3893,  0.2112,  0.0823],
        #           [ 0.4707,  0.0733,  0.1737,  ...,  0.2297,  0.0955,  0.0269],
        #           [ 0.2398, -0.1396, -0.0469,  ..., -0.0064, -0.1170, -0.0882]]]],
        #        device='cuda:0', grad_fn=<UnsqueezeBackward0>), tensor([[[[ 0.6966,  0.4594,  0.5258,  ...,  0.5017,  0.4768,  0.4122],
        #           [ 0.3601, -0.2436, -0.0931,  ..., -0.2025, -0.2776, -0.2086],
        #           [ 0.4206, -0.0857,  0.1273,  ..., -0.0504, -0.1542, -0.1370],
        #           ...,
        #           [ 0.4381, -0.0569,  0.1194,  ...,  0.0093, -0.1052, -0.1000],
        #           [ 0.3991, -0.1577, -0.0200,  ..., -0.1121, -0.1919, -0.1413],
        #           [ 0.1955, -0.2927, -0.1786,  ..., -0.2540, -0.3228, -0.2066]]]],
        #        device='cuda:0', grad_fn=<UnsqueezeBackward0>), tensor([[[[ 6.6881e-01,  4.5359e-01,  4.7008e-01,  4.2041e-01,  3.9388e-01,
        #             4.2903e-01,  4.4042e-01,  4.4056e-01,  4.4838e-01,  4.6535e-01,
        #             4.7608e-01,  4.7044e-01,  4.1334e-01,  3.1380e-01,  1.9641e-01,

        # targets
        # [BoxList(num_boxes=1, image_width=1066, image_height=800, mode=xyxy)]

        # stft_based_box
        # tensor([[[ -15.8454,  -13.3206,   38.6506,   27.2017],
        #          [ -26.3197,  -17.5163,   55.2857,   38.4621],
        #          [ -18.1193,  -17.9531,   62.9707,   39.1789],
        #          ...,
        #          [ 422.7521,  453.0185, 1278.1825, 1026.0466],
        #          [ 506.2440,  463.4796, 1440.4886, 1042.6591],
        #          [ 742.2444,  600.1472, 1357.3883,  993.6254]]], device='cuda:0')

        # stft_box_cls
        # [tensor([[[[-3.2828, -3.1959, -3.3884,  ..., -3.3873, -3.1393, -3.3094],
        #           [-3.3488, -3.3334, -3.6232,  ..., -3.6262, -3.2073, -3.0833],
        #           [-3.4302, -3.5263, -3.8737,  ..., -3.8764, -3.4635, -3.2598],
        #           ...,
        #           [-3.4276, -3.5307, -3.8785,  ..., -3.8794, -3.4669, -3.2634],
        #           [-3.1757, -3.1567, -3.4403,  ..., -3.4430, -3.1628, -3.1013],
        #           [-3.4158, -3.2729, -3.4719,  ..., -3.4738, -3.2949, -3.2474]],
        #
        #          [[-3.2318, -3.2572, -3.4265,  ..., -3.4252, -3.1989, -3.1847],
        #           [-3.0763, -3.2102, -3.5454,  ..., -3.5442, -3.1177, -3.0282],
        #           [-3.2792, -3.5201, -3.9092,  ..., -3.9144, -3.4553, -3.1968],
        #           ...,
        #           [-3.2804, -3.5267, -3.9198,  ..., -3.9191, -3.4620, -3.2066],
        #           [-3.0807, -3.1007, -3.4714,  ..., -3.4751, -3.1323, -3.1236],
        #           [-3.2583, -3.1538, -3.3486,  ..., -3.3525, -3.0854, -3.2807]]]],
        #        device='cuda:0', grad_fn=<CudnnConvolutionBackward>), tensor([[[[-3.2050, -3.1111, -3.3223,  ..., -3.3265, -3.0371, -3.2249],
        #           [-3.2759, -3.2747, -3.5872,  ..., -3.5949, -3.1304, -2.9717],
        #           [-3.3598, -3.4759, -3.8430,  ..., -3.8615, -3.4031, -3.1611],
        #           ...,

        # stft_box_reg
        # [tensor([[[[-8.5279e-02, -5.3277e-01, -3.8988e-01,  ..., -3.2037e-01,
        #            -1.6983e-01,  7.2406e-02],
        #           [-1.5083e-01, -8.4576e-01, -5.7883e-01,  ..., -4.9852e-01,
        #            -3.5223e-01, -4.3849e-02],
        #           [-1.7296e-01, -8.2091e-01, -6.1466e-01,  ..., -5.1429e-01,
        #            -3.6443e-01, -7.3191e-02],
        #           ...,
        #           [ 2.2061e-02, -4.5963e-01, -4.0323e-01,  ..., -4.8356e-01,
        #            -6.5365e-01, -3.2167e-01],
        #           [ 1.2115e-01, -3.0044e-01, -1.3809e-01,  ..., -2.3670e-01,
        #            -4.4678e-01, -1.7873e-01],
        #           [-2.3453e-01, -3.8898e-01, -2.4173e-01,  ..., -3.7911e-01,
        #            -4.7775e-01, -3.3092e-01]],
        #
        #          [[-1.6575e-01, -1.5947e-01, -1.5918e-01,  ..., -1.8277e-01,
        #            -3.1623e-02,  1.5013e-02],
        #           [-4.5310e-01, -7.2379e-01, -6.5391e-01,  ..., -5.8392e-01,
        #            -4.3005e-01, -2.8925e-01],
        #           [-3.3233e-01, -5.0810e-01, -4.9113e-01,  ..., -3.3304e-01,
        #            -1.6055e-01, -1.4407e-01],
        #           ...,
        #           [-3.2581e-01, -4.7852e-01, -6.1344e-01,  ..., -4.6545e-01,
        #            -2.8381e-01, -2.3622e-01],
        #           [-1.3952e-01, -2.1117e-01, -2.6293e-01,  ..., -1.4015e-01,
        #            -2.7761e-02, -8.9719e-02],
        #           [-3.1090e-01, -3.4540e-01, -3.8799e-01,  ..., -3.2975e-01,
        #            -2.4779e-01, -2.7028e-01]],

        losses = {
            "loss_cls": loss_cls, # tensor(0.3834, device='cuda:0', grad_fn=<DivBackward0>)
            "loss_reg": loss_box_reg, # tensor(0.3274, device='cuda:0', grad_fn=<DivBackward0>)
            "loss_centerness": loss_centerness, # tensor(0.6187, device='cuda:0', grad_fn=<DivBackward0>)
            "loss_stft_cls": loss_stft_cls, # tensor(0.3485, device='cuda:0', grad_fn=<DivBackward0>)
            "loss_stft_reg": loss_stft_reg # tensor(1.6403, device='cuda:0', grad_fn=<DivBackward0>)
        }
        return None, losses

    def _forward_test(self, shifts, box_cls, centerness, stft_box_cls, stft_box_reg, stft_based_box, image_sizes):
        boxes = self.box_selector_test(
            shifts, box_cls, centerness, stft_box_cls, stft_box_reg, stft_based_box, image_sizes)
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos_stft(cfg, in_channels):
    return STFTFCOSModule(cfg, in_channels)
