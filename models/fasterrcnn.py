import sys
sys.path.append('..')
import torch
from torch import nn
import torchvision
from typing import List, Dict, Tuple

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels: int=512):
        super().__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)

    def _clamp_bboxes(self, bboxes, img_shape):
        '''
        Clamp bounding boxes to image boundaries
        Args:
            bboxes: Tensor of shape (N, 4) containing bounding boxes
            img_shape: Tensor of shape (2,) containing the image shape
        Returns:
            Tensor of shape (N, 4) containing the clamped bounding boxes

        Local variables shape:
            bboxes: Tensor of shape (N, 4) containing the bounding boxes
        '''
        bboxes[:, 0] = torch.clamp(bboxes[:, 0], 0, img_shape[1])
        bboxes[:, 1] = torch.clamp(bboxes[:, 1], 0, img_shape[0])
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], 0, img_shape[1])
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], 0, img_shape[0])
        return bboxes

    def _bbox_deltas_to_bboxes(self, anchors: torch.Tensor, bbox_deltas: torch.Tensor) -> torch.Tensor:
        '''
        Turn anchor deltas into bounding box coordinates
        Args:
            anchors: Tensor of shape (N, 4) containing anchor boxes
            bbox_deltas: Tensor of shape (N, 4) containing bounding box deltas
        Returns:
            Tensor of shape (N, 4) containing bounding box coordinates

        Local variables shape:
            widths: Tensor of shape (N,) containing the widths of the anchors. N is the number of anchors. Value is 14 * 14 * 9 = 1764
            heights: Tensor of shape (N,) containing the heights of the anchors
            ctr_x: Tensor of shape (N,) containing the center x coordinates of the anchors
            ctr_y: Tensor of shape (N,) containing the center y coordinates of the anchors
            dx: Tensor of shape (N,) containing the x deltas
            dy: Tensor of shape (N,) containing the y deltas
            dw: Tensor of shape (N,) containing the width deltas
            dh: Tensor of shape (N,) containing the height deltas
            pred_ctr_x: Tensor of shape (N,) containing the predicted center x coordinates
            pred_ctr_y: Tensor of shape (N,) containing the predicted center y coordinates
            pred_w: Tensor of shape (N,) containing the predicted widths
            pred_h: Tensor of shape (N,) containing the predicted heights
            pred_boxes: Tensor of shape (N, 4) containing the predicted bounding boxes
        '''
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        dx = bbox_deltas[:, 0]
        dy = bbox_deltas[:, 1]
        dw = bbox_deltas[:, 2]
        dh = bbox_deltas[:, 3]
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_boxes = torch.zeros_like(anchors)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes

    def _filter_proposals(self, proposals: torch.Tensor, cls_logits: torch.Tensor, img_shape: torch.Tensor) -> torch.Tensor:
        '''
        Filter proposals based on classification scores and perform NMS
        Args:
            proposals: Tensor of shape (N, 4) containing the proposals. N = 1764
            cls_logits: Tensor of shape (N,) containing the classification logits
            img_shape: Tensor of shape (2,) containing the image shape
        Returns:
            Tensor of shape (100, 4) containing the filtered proposals
        Local variables shape:
            cls_scores: Tensor of shape (N,) containing the classification scores
            topk_indices: Tensor of shape (1000,) containing the indices of the top 1000 proposals
            proposals: Tensor of shape (1000, 4) containing the top 1000 proposals
            keep_mask: Tensor of shape (1000,) containing the mask of proposals to keep
            keep_indices: Tensor of shape (100,) containing the indices of proposals to keep
            post_nms_keep_indices: Tensor of shape (100,) containing the indices of proposals to keep after NMS
        '''
        cls_scores = torch.sigmoid(cls_logits.reshape(-1))
        _, topk_indices = torch.topk(cls_scores, k=1000)
        cls_scores = cls_scores[topk_indices]
        proposals = proposals[topk_indices]
        proposals = self._clamp_bboxes(proposals, img_shape)

        # NMS
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torchvision.ops.nms(proposals, cls_scores, iou_threshold=0.7)
        keep_mask[keep_indices] = True
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].argsort(descending=True)]

        # Keep topk proposals
        proposals = proposals[post_nms_keep_indices[:100]]
        cls_scores = cls_scores[post_nms_keep_indices[:100]]
        return proposals, cls_scores

    def _get_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Intersection over Union (IoU) of two sets of boxes.
        The box order must be (xmin, ymin, xmax, ymax).
        Args:
            boxes1: Tensor of shape (N, 4) containing the first box coordinates
            boxes2: Tensor of shape (M, 4) containing the second box coordinates
        Returns:
            Tensor of shape (N, M) containing the IoU values

        Local variables shape:
            area1: Tensor of shape (N,) containing the areas of the first boxes
            area2: Tensor of shape (M,) containing the areas of the second boxes
            lt: Tensor of shape (N, M, 2) containing the top left coordinates
            rb: Tensor of shape (N, M, 2) containing the bottom right coordinates
            wh: Tensor of shape (N, M, 2) containing the width and height
            inter: Tensor of shape (N, M) containing the intersection areas
            iou: Tensor of shape (N, M) containing the IoU values
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    def _assign_targets(self, anchors: torch.Tensor, gt_bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Assign ground truth bounding boxes to anchors
        Args:
            anchors: Tensor of shape (N, 4) containing the anchors
            gt_bboxes: Tensor of shape (M, 4) containing the ground truth bounding boxes
        Returns:
            Tuple containing:
                labels: Tensor of shape (N,) containing the labels for the anchors
                match_gt_boxes: Tensor of shape (N, 4) containing the matched ground truth bounding boxes

        Local variables shape:
            iou: Tensor of shape (N, M) containing the IoU between the anchors and ground truth bounding boxes
            best_match_iou: Tensor of shape (N,) containing the best match IoU for each anchor
            best_match_gt_index: Tensor of shape (N,) containing the index of the best match ground truth bounding box for each anchor
            best_match_gt_idx_pre_thresh: Tensor of shape (N,) containing the index of the best match ground truth bounding box for each anchor before thresholding
            below_thresh: Tensor of shape (N,) containing the mask of anchors with IoU below 0.3
            between_thresh: Tensor of shape (N,) containing the mask of anchors with IoU between 0.3 and 0.7
            match_gt_boxes: Tensor of shape (N, 4) containing the matched ground truth bounding boxes
            labels: Tensor of shape (N,) containing the labels for the anchors
            background_anchors: Tensor of shape (N,) containing the mask of background anchors
            ignored_anchors: Tensor of shape (N,) containing the mask of ignored anchors
            best_anchor_iou_for_gt: Tensor of shape (M,) containing the best anchor IoU for each ground truth bounding box
            gt_pred_pair_with_highest_iou: Tuple containing the indices of the ground truth bounding box and anchor pair with the highest IoU
            pred_inds_to_update: Tensor of shape (M,) containing the indices of the anchors to update
        '''
        if gt_bboxes.dim() == 1:
            gt_bboxes = gt_bboxes.unsqueeze(0)
        
        iou = self._get_iou(anchors, gt_bboxes)
        best_match_iou, best_match_gt_index = iou.max(dim=0)
        best_match_gt_idx_pre_thresh = best_match_gt_index.clone()

        below_thresh = best_match_iou < 0.3
        between_thresh = (best_match_iou >= 0.3) & (best_match_iou < 0.7)
        best_match_gt_index[below_thresh] = -1
        best_match_gt_index[between_thresh] = -2

        best_anchor_iou_for_gt, _ = iou.max(dim=1)
        gt_pred_pair_with_highest_iou = torch.where(iou == best_anchor_iou_for_gt[:, None])

        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_index[pred_inds_to_update] = best_match_gt_idx_pre_thresh[pred_inds_to_update]

        match_gt_boxes = gt_bboxes[best_match_gt_index.clamp(min=0, max=gt_bboxes.size(0) - 1)]
        labels = best_match_gt_index >= 0
        labels = labels.to(torch.float32)

        background_anchors = best_match_gt_index == -1
        labels[background_anchors] = 0.0

        ignored_anchors = best_match_gt_index == -2
        labels[ignored_anchors] = -1.0

        return labels, match_gt_boxes

    def _boxes_to_transformed_targets(self, gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        '''
        Convert ground truth bounding boxes to transformed targets
        Args:
            gt_boxes: Tensor of shape (N, 4) containing the ground truth bounding boxes
            anchors: Tensor of shape (N, 4) containing the anchors
        Returns:
            Tensor of shape (N, 4) containing the transformed targets

        Local variables shape:
            widths: Tensor of shape (N,) containing the widths of the ground truth bounding boxes
            heights: Tensor of shape (N,) containing the heights of the ground truth bounding boxes
            ctr_x: Tensor of shape (N,) containing the center x coordinates of the ground truth bounding boxes
            ctr_y: Tensor of shape (N,) containing the center y coordinates of the ground truth bounding boxes
            base_widths: Tensor of shape (N,) containing the widths of the anchors
            base_heights: Tensor of shape (N,) containing the heights of the anchors
            base_ctr_x: Tensor of shape (N,) containing the center x coordinates of the anchors
            base_ctr_y: Tensor of shape (N,) containing the center y coordinates of the anchors
            dx: Tensor of shape (N,) containing the x deltas
            dy: Tensor of shape (N,) containing the y deltas
            dw: Tensor of shape (N,) containing the width deltas
            dh: Tensor of shape (N,) containing the height deltas
            targets: Tensor of shape (N, 4) containing the transformed targets
        '''
        widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        ctr_x = gt_boxes[:, 0] + 0.5 * widths
        ctr_y = gt_boxes[:, 1] + 0.5 * heights

        base_widths = anchors[:, 2] - anchors[:, 0]
        base_heights = anchors[:, 3] - anchors[:, 1]
        base_ctr_x = anchors[:, 0] + 0.5 * base_widths
        base_ctr_y = anchors[:, 1] + 0.5 * base_heights

        dx = (ctr_x - base_ctr_x) / base_widths
        dy = (ctr_y - base_ctr_y) / base_heights
        dw = torch.log(widths / base_widths)
        dh = torch.log(heights / base_heights)

        targets = torch.stack((dx, dy, dw, dh), dim=1)
        return targets

    def _sample_positive_negative_anchors(self, labels: torch.Tensor, positive_count: int, total_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Sample positive and negative anchors to train on
        Args:
            labels: Tensor of shape (N,) containing the labels for the anchors
            positive_count: Number of positive anchors to sample
            total_count: Total number of anchors to sample
        Returns:
            Tuple containing:
                sampled_pos_idx_mask: Tensor of shape (N,) containing the mask of sampled positive indices
                sampled_neg_idx_mask: Tensor of shape (N,) containing the mask of sampled negative indices

        Local variables shape:
            positive_indices: Tensor of shape (N,) containing the indices of positive anchors
            negative_indices: Tensor of shape (N,) containing the indices of negative anchors
            num_pos: Number of positive anchors to sample
            num_neg: Number of negative anchors to sample
            perm_pos_idx: Tensor of shape (num_pos,) containing the permuted positive indices
            perm_neg_idx: Tensor of shape (num_neg,) containing the permuted negative indices
            pos_idx: Tensor of shape (num_pos,) containing the positive indices to sample
            neg_idx: Tensor of shape (num_neg,) containing the negative indices to sample
            sampled_pos_idx_mask: Tensor of shape (N,) containing the mask of sampled positive indices
            sampled_neg_idx_mask: Tensor of shape (N,) containing the mask of sampled negative
        '''
        positive_indices = torch.where(labels >= 1)[0]
        negative_indices = torch.where(labels == 0)[0]
        num_pos = min(positive_count, positive_indices.numel())
        num_neg = min(total_count - num_pos, negative_indices.numel())
        perm_pos_idx = torch.randperm(positive_indices.numel(), device=labels.device)[:num_pos]
        perm_neg_idx = torch.randperm(negative_indices.numel(), device=labels.device)[:num_neg]
        pos_idx = positive_indices[perm_pos_idx]
        neg_idx = negative_indices[perm_neg_idx]
        sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
        sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
        sampled_pos_idx_mask[pos_idx] = True
        sampled_neg_idx_mask[neg_idx] = True
        return sampled_pos_idx_mask, sampled_neg_idx_mask

    def _generate_anchors(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        '''
        Generate anchors for the image
        Args:
            img: Tensor of shape (N, C, H, W) containing the image. The N doesn't matter since we only need the height and width
            feat: Tensor of shape (N, C, H, W) containing the feature map.
        Returns:
            Tensor of shape (N * grid_h * grid_w * num_anchors, 4) containing the anchors.

        Local variables shape:
            grid_h: Height of the feature map, shape = (1,). Value is 14
            grid_w: Width of the feature map, shape = (1,). Value is 14
            img_h: Height of the image, shape = (1,). Value is 224
            img_w: Width of the image, shape = (1,). Value is 224
            stride_h: Tensor of shape (1,) containing the stride height
            stride_w: Tensor of shape (1,) containing the stride width
            scales: Tensor of shape (num_scales,) containing the scales. num_scales = 3
            aspect_ratios: Tensor of shape (num_aspect_ratios,) containing the aspect ratios. num_aspect_ratios = 3
            h_ratios: Tensor of shape (num_aspect_ratios,) containing the height ratios
            w_ratios: Tensor of shape (num_aspect_ratios,) containing the width ratios
            ws: Tensor of shape (num_scales * num_aspect_ratios,) containing the width ratios for each scale and aspect ratio
            hs: Tensor of shape (num_scales * num_aspect_ratios,) containing the height ratios for each scale and aspect ratio
            base_anchors: Tensor of shape (num_scales * num_aspect_ratios, 4) containing the base anchors
            shifts_x: Tensor of shape (grid_w,) containing the x shifts
            shifts_y: Tensor of shape (grid_h,) containing the y shifts
            shifts: Tensor of shape (grid_h * grid_w, 4) containing the shifts
            anchors: Tensor of shape (grid_h * grid_w * num_anchors, 4) containing the anchors. Num anchors = 14 * 14 * 9 = 1764
        '''
        grid_h, grid_w = feat.shape[-2:]
        img_h, img_w = img.shape[-2:]
        stride_h = torch.tensor(img_h / grid_h, dtype=torch.int64, device=img.device)
        stride_w = torch.tensor(img_w / grid_w, dtype=torch.int64, device=img.device)
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=img.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=img.device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=img.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=img.device) * stride_h
        shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_y, indexing='ij')
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
        return anchors
    
    def forward(self, 
                imgs: torch.Tensor, 
                feats: torch.Tensor, 
                targets: List[Dict[str, torch.Tensor]]=None) -> Dict[str, torch.Tensor]:
        '''
        Forward pass of the Region Proposal Network
        Args:
            imgs: Tensor of shape (N, C, H, W) containing the images
            feats: Tensor of shape (N, C, H, W) containing the feature maps
            targets: List of dictionaries containing the target bounding boxes and labels
        Returns:
            Dictionary containing the classification and bounding box regression losses

        Local variables shape:
            batch_size: Shape of the batch, shape = (1,). Value is 2
            loss_dict: Dictionary containing the classification and bounding box regression losses
            proposals: List containing the proposals for each image
            img_idx: Index of the image
            feat: Tensor of shape (1, C, H, W) containing the feature map
            img: Tensor of shape (1, C, H, W) containing the image
            anchors: Tensor of shape (1764, 4) containing the anchors
            cls_logits: Tensor of shape (1764, 1) containing the classification logits
            bbox_preds: Tensor of shape (1764, 4) containing the bounding box predictions
            proposals_per_img: Tensor of shape (1764, 4) containing the proposals for the image
            scores_per_img: Tensor of shape (1764,) containing the classification scores for the image
            gt_bboxes: Tensor of shape (M, 4) containing the ground truth bounding boxes
            labels: Tensor of shape (1764,) containing the labels for the anchors
            matched_gt_boxes: Tensor of shape (1764, 4) containing the matched ground truth bounding boxes
            sampled_pos_idx_mask: Tensor of shape (1764,) containing the mask of sampled positive indices
            sampled_neg_idx_mask: Tensor of shape (1764,) containing the mask of sampled negative indices
            sampled_idx_mask: Tensor of shape (256,) containing the mask of sampled indices
            bbox_preds: Tensor of shape (256, 4) containing the bounding box predictions
            bbox_targets: Tensor of shape (256, 4) containing the bounding box targets
            cls_loss: Tensor of shape (1,) containing the classification loss
            bbox_loss: Tensor of shape (1,) containing the bounding box regression loss
        '''
        batch_size = feats.size(0)
        loss_dict = {'cls_loss': [], 'bbox_loss': []}
        proposals = []
        for img_idx in range(batch_size):
            feat = feats[img_idx:img_idx+1]
            img = imgs[img_idx:img_idx+1]
            anchors = self._generate_anchors(img, feat)
            feat = nn.functional.relu(self.conv(feat))
            cls_logits = self.cls_layer(feat).permute(0, 2, 3, 1).reshape(-1, 1)
            bbox_preds = self.bbox_reg_layer(feat).permute(0, 2, 3, 1).reshape(-1, 4)
            proposals_per_img = self._bbox_deltas_to_bboxes(anchors, bbox_preds)
            proposals_per_img, scores_per_img = self._filter_proposals(proposals_per_img, cls_logits, img.shape[-2:])
            proposals.append(proposals_per_img)

            if targets is not None:
                gt_bboxes = targets[img_idx]['bboxes']
                labels, matched_gt_boxes = self._assign_targets(anchors, gt_bboxes)
                sampled_pos_idx_mask, sampled_neg_idx_mask = self._sample_positive_negative_anchors(
                    labels, positive_count=128, total_count=256
                )
                sampled_idx_mask = sampled_pos_idx_mask | sampled_neg_idx_mask
                labels = labels[sampled_idx_mask]
                sampled_idx_mask = sampled_idx_mask.nonzero(as_tuple=True)[0]
                bbox_preds = bbox_preds[sampled_idx_mask]
                matched_gt_boxes = matched_gt_boxes[sampled_idx_mask]
                bbox_targets = self._boxes_to_transformed_targets(matched_gt_boxes, anchors[sampled_idx_mask])
                cls_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits[sampled_idx_mask], labels.unsqueeze(1))
                bbox_loss = nn.functional.smooth_l1_loss(bbox_preds, bbox_targets)
                loss_dict['cls_loss'].append(cls_loss)
                loss_dict['bbox_loss'].append(bbox_loss)

        loss_dict['cls_loss'] = torch.stack(loss_dict['cls_loss']).mean()
        loss_dict['bbox_loss'] = torch.stack(loss_dict['bbox_loss']).mean()

        return loss_dict, proposals