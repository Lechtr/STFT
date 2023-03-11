import os
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from stft_core.modeling.utils import cat
from stft_core.structures.bounding_box import BoxList
from stft_core.structures.boxlist_ops import boxlist_iou


INF = 100000000

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def sigmoid_focal_loss(
    # show the first 10 items of the arrays in the following comments
    inputs, # -4.9781 -5.0811, -5.3265 -5.4008, -5.3326 -5.4104, -5.3522 -5.4352, -5.3846 -5.4699, -5.4182 -5.5032, -5.4411 -5.5264, -5.4531 -5.5391, -5.4581 -5.5448, -5.4600 -5.5474
    targets, # 0  0, 0  0, 0  0, 0  0, 0  0, 0  0, 0  0, 0  0, 0  0, 0  0
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    # BCEWithLogitsLoss(inputs, targets, reduction="none")
    # torchvision.ops.sigmoid_focal_loss()

    # original version
    p = torch.sigmoid(inputs) # 0.001 *: 6.8632  6.1936,  4.8491  4.5029,  4.8197  4.4600,  4.7265  4.3510,  4.5760  4.2027,  4.4253  4.0652,  4.3252  3.9723,  4.2739  3.9225,  4.2524  3.9001,  4.2444  3.8901
    # ce_loss = F.binary_cross_entropy_with_logits(
    #     inputs, targets, reduction="none"
    # ) # 0.001 *: 6.8398  6.1745, 4.8374  4.4927, 4.8081  4.4501, 4.7155  4.3415, 4.5655  4.1939, 4.4154  4.0570, 4.3159  3.9643, 4.2648  3.9148, 4.2434  3.8924, 4.2353  3.8826

    # multiclass loss
    # computes better with targets as int of class 0, 1 or 2 instead of one-hot-encoded
    ce_loss = CrossEntropyLoss(inputs, targets, reduction="none")


    p_t = p * targets + (1 - p) * (1 - targets) # 0.9932  0.9938, 0.9952  0.9955, 0.9952  0.9955, 0.9953  0.9957, 0.9954  0.9958, 0.9956  0.9959, 0.9957  0.9960, 0.9957  0.9961, 0.9958  0.9961, 0.9958  0.9961
    loss = ce_loss * ((1 - p_t) ** gamma) # 1e-07 *: 3.2108  2.3613, 1.1347  0.9089, 1.1142  0.8832, 1.0510  0.8201, 0.9538  0.7392, 0.8628  0.6691, 0.8057  0.6243, 0.7774  0.6011, 0.7657  0.5909, 0.7614  0.586,
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500, 0.7500  0.7500
        loss = alpha_t * loss # 1e-07 : ,3.1462  2.2038, 1.3251  0.9630, 1.2719  0.9041, 1.1880  0.8279, 1.0954  0.7552, 1.0236  0.7017, 0.9821  0.6673, 0.9632  0.6492, 0.9551  0.6386, 0.9479  0.6317

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss # 2.49932

# sigmoid_focal_loss_jit = torch.jit.script(
#     sigmoid_focal_loss
# )  # type: torch.jit.ScriptModule

def iou_loss(inputs, targets, weight=None, box_mode="xyxy", loss_type="iou", reduction="none"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss

def smooth_l1_loss(input,
                   target,
                   beta: float,
                   reduction: str = "none",
                   size_average=False):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if reduction == "mean" or size_average:
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def permute_all_cls_and_box_to_N_HWA_K_and_concat(
    box_cls, box_delta, box_center, stft_box_cls, stft_box_delta, num_classes=2):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)

    if len(stft_box_cls)!=0:
        stft_box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in stft_box_cls]
        stft_box_cls = cat(stft_box_cls_flattened, dim=1).view(-1, num_classes)

    if len(stft_box_delta)!=0:
        stft_box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in stft_box_delta]
        stft_box_delta = cat(stft_box_delta_flattened, dim=1).view(-1, 4)
    
    return box_cls, box_delta, box_center, stft_box_cls, stft_box_delta


class Shift2BoxTransform(object):
    def __init__(self, weights):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dl, dt, dr, db) deltas.
        """
        self.weights = weights

    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.

        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts),
                           dim=-1) * shifts.new_tensor(self.weights)
        return deltas

    def apply_deltas(self, deltas, shifts):
        """
        Apply transformation `deltas` (dl, dt, dr, db) to `shifts`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single shift shifts[i].
            shifts (Tensor): shifts to transform, of shape (N, 2)
        """
        assert torch.isfinite(deltas).all().item()
        shifts = shifts.to(deltas.dtype)

        if deltas.numel() == 0:
            return torch.empty_like(deltas)

        deltas = deltas.view(deltas.size()[:-1] + (-1, 4)) / shifts.new_tensor(self.weights)
        boxes = torch.cat((shifts.unsqueeze(-2) - deltas[..., :2],
                           shifts.unsqueeze(-2) + deltas[..., 2:]),
                          dim=-1).view(deltas.size()[:-2] + (-1, ))
        return boxes




class STFTFCOSLossComputation(object):
    """
    This class computes the STFTFCOS losses.
    """
    def __init__(self, cfg):
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        self.shift2box_transform = Shift2BoxTransform(
            weights=(1.0, 1.0, 1.0, 1.0))
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.stft_iou_thresh = cfg.MODEL.STFT.IOU_THRESH
        self.stft_bbox_std = cfg.MODEL.STFT.BBOX_STD
        self.stft_reg_beta = cfg.MODEL.STFT.REG_BETA

    @torch.no_grad()
    def get_ground_truth(self,
                         shifts, # shape [1, 5, 12800, 2]
                         targets, # [BoxList(num_boxes=1, image_width=1020, image_height=800, mode=xyxy)]
                         pre_boxes_list # shape: [1, 17064, 4]
                         ):
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        stft_gt_classes = []
        stft_gt_shifts_deltas = []

        for shifts_per_image, targets_per_image, pre_boxes in zip(shifts, targets, pre_boxes_list): # shifts_per_image: [5, 12800, 2], pre_boxes: [17064, 4]
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.size(0), -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0) # tensor([[-1.0000e+00,  6.4000e+01], [-1.0000e+00,  6.4000e+01], ..., [ 5.1200e+02,  1.0000e+08]], device='cuda:0'); shape: [17064, 2]

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0) # tensor([[  4.,   4.], [ 12.,   4.], [ 20.,   4.], ..., [704., 832.], [832., 832.], [960., 832.]], device='cuda:0'); torch.Size([17064, 2])

            gt_boxes = targets_per_image.bbox # tensor([[496.9392, 267.2414, 765.7230, 520.6896]], device='cuda:0')
            deltas = self.shift2box_transform.get_deltas(shifts_over_all_feature_maps, gt_boxes.unsqueeze(1)) # tensor([[[-492.9392, -263.2414,  761.7230,  516.6896], [-484.9392, -263.2414,  753.7230,  516.6896], [-476.9392, -263.2414,  745.7230,  516.6896], ..., [ 207.0608,  564.7587,   61.7230, -311.3104], [ 335.0608,  564.7587,  -66.2770, -311.3104], [ 463.0608,  564.7587, -194.2770, -311.3104]]], device='cuda:0'); torch.Size([1, 17064, 4])

            if self.center_sampling_radius > 0:
                centers = targets_per_image.center() # tensor([[631.3311, 393.9655]], device='cuda:0')
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes[:, :2]),
                        torch.min(centers + radius, gt_boxes[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = targets_per_image.area().unsqueeze(1).repeat(1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = INF
            gt_positions_area[~is_cared_in_the_level] = INF

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)
            # positions_min_area: tensor([100000000., 100000000.,  ...,,100000000.], device='cuda:0'); torch.Size([17064])
            # gt_matched_idxs: tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'); torch.Size([17064]), alles 0
            # positions_min_area[positions_min_area<100000000]: tensor([68646.0078, 68646.0078, 68646.0078, 68646.0078, 68646.0078, 68646.0078, 68646.0078, 68646.0078, 68646.0078], device='cuda:0')

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, targets_per_image[gt_matched_idxs].bbox)

            # ground truth classes
            labels_per_im = targets_per_image.get_field("labels") # tensor([2], device='cuda:0')
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = labels_per_im[gt_matched_idxs] # tensor([2, 2, 2,  ..., 2, 2, 2], device='cuda:0')
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == INF] = self.num_classes+5 #for not gt # torch.Size([17055])
                # alles 7er bis auf gt_classes_i[gt_classes_i<7]: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2], device='cuda:0'); torch.Size([17064])
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs)+self.num_classes+5 #for not gt

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)


            # stft
            iou = boxlist_iou(BoxList(pre_boxes, targets_per_image.size, targets_per_image.mode), targets_per_image) # tensor([[0.0000], [0.0000], [0.0000], ..., [0.0373], [0.0311], [0.0000]], device='cuda:0'); torch.Size([17064, 1])
            (max_iou, argmax_iou) = iou.max(dim=1)
            # max_iou: tensor([0.0000, 0.0000, 0.0000,  ..., 0.0373, 0.0311, 0.0000], device='cuda:0'), torch.Size([17064]), zwischen 0 und 0.7139
            # argmax_iou: tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'); torch.Size([17064]); alles 0
            invalid = max_iou < self.stft_iou_thresh # 0.1
            #invalid: tensor([True, True, True,  ..., True, True, True], device='cuda:0'); torch.Size([17064]); sum(invalid): tensor(15878, device='cuda:0')
            gt_target = gt_boxes[argmax_iou] # tensor([[496.9392, 267.2414, 765.7230, 520.6896], [496.9392, 267.2414, 765.7230, 520.6896], [496.9392, 267.2414, 765.7230, 520.6896], ...,

            stft_cls_target = labels_per_im[argmax_iou] # tensor([2, 2, 2,  ..., 2, 2, 2], device='cuda:0'); torch.Size([17064])
            stft_cls_target[invalid] = self.num_classes+5 #for not gt
            # stft_cls_target: tensor([7, 7, 7,  ..., 7, 7, 7], device='cuda:0'); torch.Size([17064])
            # stft_cls_target[stft_cls_target<7].shape: torch.Size([1186])

            stft_bbox_std = pre_boxes.new_tensor(self.stft_bbox_std)
            pre_boxes_wh = pre_boxes[:, 2:4] - pre_boxes[:, 0:2]
            pre_boxes_wh = torch.cat([pre_boxes_wh, pre_boxes_wh], dim=1)
            stft_off_target = (gt_target - pre_boxes) / (pre_boxes_wh * stft_bbox_std)

            stft_gt_classes.append(stft_cls_target)
            stft_gt_shifts_deltas.append(stft_off_target)

        return (
            torch.stack(gt_classes),
            torch.stack(gt_shifts_deltas),
            torch.stack(gt_centerness),
            torch.stack(stft_gt_classes),
            torch.stack(stft_gt_shifts_deltas),
        )


    def __call__(self, shifts,
                 pred_class_logits, # box_cls
                 pred_shift_deltas, # box_regression
                 pred_centerness, # centerness
                 targets, # targets
                 bd_based_box, # stft_based_box
                 stft_bbox_cls, # stft_box_cls
                 stft_bbox_reg # stft_box_reg
                 ):

        (
            gt_classes, # tensor([[7, 7, 7,  ..., 7, 7, 7]], device='cuda:0'), len(): 16289
            gt_shifts_deltas, # tensor([[[-418.0276, -378.1429,  683.9816,  658.5000],
                              # [-410.0276, -378.1429,  675.9816,  658.5000],
                              # [-402.0276, -378.1429,  667.9816,  658.5000],
                              # ...,
                              # [ 281.9724,  449.8571,  -16.0184, -169.5000],
                              # [ 409.9724,  449.8571, -144.0184, -169.5000],
                              # [ 537.9724,  449.8571, -272.0184, -169.5000]]], device='cuda:0')
            gt_centerness, # tensor([[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
            stft_gt_classes, # tensor([[7, 7, 7,  ..., 2, 2, 7]], device='cuda:0')
            stft_gt_shifts_deltas, # tensor([[[17.0964, 19.6779, 25.4766, 31.6015],
                                   # [11.7800, 14.4907, 16.8000, 22.6185],
                                   # [11.6342, 14.2929, 16.6973, 22.2759],
                                   # ...,
                                   # [ 0.3030, -0.1358, -0.9936, -1.1743],
                                   # [ 0.1181, -0.2530, -1.2688, -1.2617],
                                   # [-0.7409, -1.2328, -1.8114, -1.6853]]], device='cuda:0')
        ) = self.get_ground_truth(shifts, targets, bd_based_box)

        (
            pred_class_logits, # tensor([[-4.8728, -5.1408],
                                # [-5.2613, -5.5084],
                                # [-5.2732, -5.5276],
                                # ...,
                                # [-3.6198, -3.9084],
                                # [-4.0312, -4.3180],
                                # [-3.8618, -4.8286]], device='cuda:0', grad_fn=<ViewBackward>)
            pred_shift_deltas, #tensor([[ 20.0040,  18.0807,  31.2385,  22.1902],
                                # [ 36.2581,  22.9876,  39.5121,  32.3763],
                                # [ 36.0427,  23.2489,  39.2647,  32.9178],
                                # ...,
                                # [396.5472, 410.2602, 359.6518, 172.9366],
                                # [461.2139, 378.3073, 406.3593, 187.3258],
                                # [325.9882, 226.5166, 246.2104, 135.8087]], device='cuda:0',
                                # grad_fn=<ViewBackward>)
            pred_centerness,
            stft_class_logits, # tensor([[-3.4625, -3.4591],
                                # [-3.3978, -3.4696],
                                # [-3.5369, -3.5909],
                                # ...,
                                # [-1.4151, -1.0552],
                                # [-1.6657, -1.3548],
                                # [-2.0824, -2.0467]], device='cuda:0', grad_fn=<ViewBackward>)
            stft_shift_deltas
        ) = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_shift_deltas, pred_centerness,
            stft_bbox_cls, stft_bbox_reg,
            self.num_classes # 2
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        # fcos
        gt_classes = gt_classes.flatten().long() # tensor([7, 7, 7,  ..., 7, 7, 7], device='cuda:0'); torch.Size([17064])
        # gt_classes[gt_classes<7]: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], device='cuda:0')
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = gt_classes >= 0 # tensor([True, True, True,  ..., True, True, True], device='cuda:0'); torch.Size([17064])
        foreground_idxs = (gt_classes >= 0) & (gt_classes != (self.num_classes+5)) # tensor([False, False, False,  ..., False, False, False], device='cuda:0'); torch.Size([17064]);
        num_foreground = foreground_idxs.sum() # tensor(15, device='cuda:0')
        acc_centerness_num = gt_centerness[foreground_idxs].sum() # tensor(10.6697, device='cuda:0')

        # original
        # gt_classes_target = torch.zeros_like(pred_class_logits) # torch.Size([17064, 2])
        # gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]-1] = 1

        # Integer-Class_index statt one-hot-encoded
        gt_classes_target = torch.zeros(pred_class_logits.shape[0], device='cuda:0').long()  # torch.Size([17064])
        gt_classes_target[foreground_idxs] = gt_classes[foreground_idxs] -1

        num_gpus = get_num_gpus()
        num_foreground_avg_per_gpu = max(reduce_sum(num_foreground).item() / float(num_gpus), 1.0)
        acc_centerness_num_avg_per_gpu = max(reduce_sum(acc_centerness_num).item() / float(num_gpus), 1.0)

        # logits loss
        loss_cls = sigmoid_focal_loss(
            pred_class_logits[valid_idxs], # tensor([[-4.8728, -5.1408],
                                            # [-5.2613, -5.5084],
                                            # [-5.2732, -5.5276],
                                            # ...,
            gt_classes_target[valid_idxs], # tensor([[0., 0.], # max is 1
                                            # [0., 0.],
                                            # [0., 0.],
                                            # ...,
            alpha=self.focal_loss_alpha, # 0.25
            gamma=self.focal_loss_gamma, # 2.0
            reduction="sum",
        ) / num_foreground_avg_per_gpu # 9.0

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / acc_centerness_num_avg_per_gpu

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / num_foreground_avg_per_gpu


        # stft
        stft_gt_classes = stft_gt_classes.flatten().long() # tensor([7, 7, 7,  ..., 7, 7, 7], device='cuda:0')
        stft_gt_shifts_deltas = stft_gt_shifts_deltas.view(-1, 4) # tensor([[13.6345,  7.3209, 18.1021, 20.8582], [ 9.3808,  5.4616, 11.7169, 14.6950], [ 9.2391,  5.3799, 11.6057, 14.4191], ..., [ 0.1688, -1.1297, -1.4255, -2.0119], [-0.0473, -1.2312, -1.6757, -2.0927], [-0.8740, -2.5119, -2.3062, -2.8309]], device='cuda:0')

        valid_idxs_stft = stft_gt_classes >= 0. # tensor([True, True, True,  ..., True, True, True], device='cuda:0')
        foreground_idxs_stft = (stft_gt_classes >= 0) & (stft_gt_classes != (self.num_classes+5)) # tensor([False, False, False,  ...,  True,  True, False], device='cuda:0')
        num_foreground_stft = foreground_idxs_stft.sum() # tensor(788, device='cuda:0')
        num_foreground_stft = max(reduce_sum(num_foreground_stft).item() / float(num_gpus), 1.0) # 788.0

        # original
        # stft_gt_classes_target = torch.zeros_like(stft_class_logits) # tensor([[0., 0.],... # shape: torch.Size([17884, 2])
        # stft_gt_classes_target[foreground_idxs_stft, stft_gt_classes[foreground_idxs_stft]-1] = 1

        # Integer-Class_index statt one-hot-encoded
        stft_gt_classes_target = torch.zeros(stft_class_logits.shape[0], device='cuda:0').long()  # torch.Size([17064, 2])
        stft_gt_classes_target[foreground_idxs_stft] = stft_gt_classes[foreground_idxs_stft] - 1

        loss_stft_cls = sigmoid_focal_loss(
            stft_class_logits[valid_idxs_stft], # tensor([[-3.4625, -3.4591], [-3.3978, -3.4696], [-3.5369, -3.5909], ..., [-1.4151, -1.0552], [-1.6657, -1.3548], [-2.0824, -2.0467]], device='cuda:0', grad_fn=<IndexBackward>)
            stft_gt_classes_target[valid_idxs_stft], # tensor([[0., 0.], [0., 0.], [0., 0.], ..., [0., 1.], [0., 1.], [0., 0.]], device='cuda:0')
            alpha=self.focal_loss_alpha, # 0.25
            gamma=self.focal_loss_gamma, # 2.0
            reduction="sum",
        ) / num_foreground_stft

        if foreground_idxs_stft.numel() > 0: # 16289
            loss_stft_reg = (
                smooth_l1_loss(
                    stft_shift_deltas[foreground_idxs_stft],
                    stft_gt_shifts_deltas[foreground_idxs_stft],
                    beta=self.stft_reg_beta,
                    reduction="sum"
                ) / num_foreground_stft
            )
        else:
            loss_stft_reg = stft_shift_deltas.sum()

        return loss_cls, loss_box_reg, loss_centerness, loss_stft_cls, loss_stft_reg


def make_fcos_stft_loss_evaluator(cfg):
    loss_evaluator = STFTFCOSLossComputation(cfg)
    return loss_evaluator
