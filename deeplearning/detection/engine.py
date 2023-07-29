import math
import sys

import numpy as np

import torch
import torchvision.models.detection.mask_rcnn
from . import utils
#from .coco_eval import CocoEvaluator
#from .coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            # loss_dict['loss_keypoint'].backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    loss_fn = torch.nn.MSELoss()
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)
    oks_values = []
    running_loss = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        
        losses = []
        for i in range(len(outputs)):
            
            loss_keypoint = loss_fn(outputs[i]['keypoints'][0], targets[i]['keypoints'][0])
            losses.append(loss_keypoint.item())
        running_loss += sum(losses) / len(losses)
        
        for i_batch in range(len(targets)):
            oks = compute_oks(targets[0]['keypoints'], outputs[0]['keypoints'])
            oks_values.append(oks)
    metric_logger.update(loss=running_loss)    

    oks_values = np.array(oks_values)
    print("validation oks:" , oks_values)
    
    return metric_logger,oks_values
    
    

def compute_oks(target, pred, sigmas=None, in_vis_thre=None):
    """
    Compute Object Keypoint Similarity (OKS) between predicted keypoints and ground truth keypoints.
    Return oks per keypoint class
    OKS ranges from 0 to 1, with 1 as the best prediction and 0 as the worst predictions
    """
    # Prepare sigmas and in_vis_thre if not provided
    
    target = target[0].detach().numpy()
    if pred.shape[0]==0:
        print("prediction doesn't have keypoints")
        return np.zeros((target.shape[0],))
    pred=pred[0].detach().numpy()
    if sigmas is None:
        # sigmas = [.026, .025, .025, .035, .035,
        #           .079, .079, .072, .072, .062,
        #           .062, .107, .107, .087, .087,
        #           .089, .089]
        
        sigmas = [1] * target.shape[0]
    if in_vis_thre is None:
        in_vis_thre = 0.0

    # Create an array to store the OKS values
    oks = np.zeros((len(sigmas),))

    # Get the visible keypoints in the target
    vis_target = target[:,2] > in_vis_thre

    # For each visible keypoint, compute the OKS
    for i in range(len(sigmas)):
        # print(pred[i,:2], target[i,:2])
        d = np.linalg.norm(pred[i,:2] - target[i,:2])
        e = d / sigmas[i]
        oks[i] = np.exp(-e**2 / 2) if vis_target[i] else 1

    return oks




#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator

    

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# @torch.inference_mode()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for images, targets in metric_logger.log_every(data_loader, 100, header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator
