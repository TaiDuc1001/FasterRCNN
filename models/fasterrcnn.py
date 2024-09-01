import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from typing import List, Tuple, Union
from models.utils import *

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(weights="DEFAULT")
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        
    def forward(self, img_data: torch.Tensor) -> torch.Tensor:
        return self.backbone(img_data)
    
class ProposalModule(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 512, n_anchors: int = 9, p_dropout: float = 0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)
        
    def forward(self, 
                feature_map: torch.Tensor, 
                pos_anc_ind=None, 
                neg_anc_ind=None, 
                pos_anc_coords=None) -> Union[
                    Tuple[torch.Tensor, torch.Tensor], 
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        '''
        Forward pass for the proposal module
        Args:
            feature_map: torch.Tensor - input feature map from the backbone
            pos_anc_ind: torch.Tensor - indices of positive anchors
            neg_anc_ind: torch.Tensor - indices of negative anchors
            pos_anc_coords: torch.Tensor - coordinates of positive anchors
        Returns:
            Tuple[torch.Tensor, torch.Tensor] - conf_scores_pred, reg_offsets_pred
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] - conf_scores_pos, conf_scores_neg, offsets_pos, proposals
        Local variables shape:
            out: torch.Tensor shape (B, hidden_dim, hmap, wmap) - output of the first convolutional layer
            reg_offsets_pred: torch.Tensor shape (B, A*4, hmap, wmap) - predicted offsets for anchors
            conf_scores_pred: torch.Tensor shape (B, A, hmap, wmap) - predicted conf scores for anchors
            conf_scores_pos: torch.Tensor shape (N, ) - conf scores for positive anchors
            conf_scores_neg: torch.Tensor shape (N, ) - conf scores for negative anchors
            offsets_pos: torch.Tensor shape (N, 4) - offsets for positive anchors
            proposals: torch.Tensor shape (N, 4) - proposals generated using offsets
        '''
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'
            
        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))
        
        reg_offsets_pred = self.reg_head(out) # (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out) # (B, A, hmap, wmap)
        
        if mode == 'train': 
            # get conf scores 
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            # get offsets for +ve anchors
            offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
            # generate proposals using offsets
            proposals = generate_proposals(pos_anc_coords, offsets_pos)
            
            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals
            
        elif mode == 'eval':
            return conf_scores_pred, reg_offsets_pred
        
class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size: Tuple[int, int], out_size: Tuple[int, int], out_channels: int):
        super().__init__()
        
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size
        
        # downsampling scale factor 
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h 
        
        # scales and ratios for anchor boxes
        self.anc_scales = [2, 4, 6]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)
        
        # IoU thresholds for +ve and -ve anchors
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        
        # weights for loss
        self.w_conf = 1
        self.w_reg = 5
        
        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels, n_anchors=self.n_anc_boxes)
        
    def forward(self, 
                images: torch.Tensor, 
                gt_bboxes: torch.Tensor, 
                gt_classes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the Region Proposal Network
        Args:
            images: torch.Tensor - input images
            gt_bboxes: torch.Tensor - ground truth bounding boxes
            gt_classes: torch.Tensor - ground truth classes
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] - total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos
        Local variables shape:
            batch_size: int - number of samples in the batch
            feature_map: torch.Tensor shape (B, C, H, W) - output feature map from the backbone
            anc_pts_x: torch.Tensor shape (W, ) - x-coordinates of anchor points
            anc_pts_y: torch.Tensor shape (H, ) - y-coordinates of anchor points
            anc_base: torch.Tensor shape (A, 4, H, W) - base anchor boxes
            anc_boxes_all: torch.Tensor shape (B, A, 4, H, W) - all anchor boxes for each image
            gt_bboxes_proj: torch.Tensor shape (B, N, 4) - projected ground truth boxes
            positive_anc_ind: torch.Tensor shape (N, ) - indices of positive anchors
            negative_anc_ind: torch.Tensor shape (N, ) - indices of negative anchors
            GT_conf_scores: torch.Tensor shape (N, ) - ground truth confidence scores
            GT_offsets: torch.Tensor shape (N, 4) - ground truth offsets
            GT_class_pos: torch.Tensor shape (N, ) - ground truth classes for positive anchors
            positive_anc_coords: torch.Tensor shape (N, 4) - coordinates of positive anchors
            negative_anc_coords: torch.Tensor shape (N, 4) - coordinates of negative anchors
            positive_anc_ind_sep: torch.Tensor shape (N, ) - indices of positive anchors separated by image
            conf_scores_pos: torch.Tensor shape (N, ) - confidence scores for positive anchors
            conf_scores_neg: torch.Tensor shape (N, ) - confidence scores for negative anchors
            offsets_pos: torch.Tensor shape (N, 4) - offsets for positive anchors
            proposals: torch.Tensor shape (N, 4) - proposals generated using offsets
            cls_loss: torch.Tensor - classification loss
            reg_loss: torch.Tensor - regression loss
            total_rpn_loss: torch.Tensor - total RPN loss
        '''
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images)
        
        # generate anchors
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        
        # get positive and negative anchors amongst other things
        gt_bboxes_proj = project_bboxes(gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')
        
        positive_anc_ind, negative_anc_ind, GT_conf_scores, \
        GT_offsets, GT_class_pos, positive_anc_coords, \
        negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes)
        
        # pass through the proposal module
        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(feature_map, positive_anc_ind, \
                                                                                        negative_anc_ind, positive_anc_coords)
        
        cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
        reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
        
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        
        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos
    
    def inference(self, 
                  images: torch.Tensor, 
                  conf_thresh: float = 0.5, 
                  nms_thresh: float = 0.7) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        '''
        Inference for the Region Proposal Network
        Args:
            images: torch.Tensor - input images
            conf_thresh: float - confidence threshold
            nms_thresh: float - NMS threshold
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]] - proposals_final, conf_scores_final, feature_map
        Local variables shape:
            batch_size: int - number of samples in the batch
            feature_map: torch.Tensor shape (B, C, H, W) - output feature map from the backbone
            anc_pts_x: torch.Tensor shape (W, ) - x-coordinates of anchor points
            anc_pts_y: torch.Tensor shape (H, ) - y-coordinates of anchor points
            anc_base: torch.Tensor shape (A, 4, H, W) - base anchor boxes
            anc_boxes_all: torch.Tensor shape (B, A, 4, H, W) - all anchor boxes for each image
            conf_scores_pred: torch.Tensor shape (B, A, H, W) - predicted conf scores for anchors
            offsets_pred: torch.Tensor shape (B, A*4, H, W) - predicted offsets for anchors
            proposals_final: List[torch.Tensor] - final proposals for each image
            conf_scores_final: List[torch.Tensor] - confidence scores for each proposal
        '''
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)

            # generate anchors
            anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
            anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

            # get conf scores and offsets
            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            # filter out proposals based on conf threshold and nms threshold for each image
            proposals_final = []
            conf_scores_final = []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                proposals = generate_proposals(anc_boxes, offsets)
                # filter based on confidence threshold
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                # filter based on nms threshold
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)
            
        return proposals_final, conf_scores_final, feature_map
    
class ClassificationModule(nn.Module):
    def __init__(self, 
                 out_channels: int, 
                 n_classes: int, 
                 roi_size: Tuple[int, int], 
                 hidden_dim: int = 1024, 
                 p_dropout: float = 0.3):
        super().__init__()        
        self.roi_size = roi_size
        # hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        
        # define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, 
                feature_map: torch.Tensor, 
                proposals_list: List[torch.Tensor], 
                gt_classes: torch.Tensor = None) -> Union[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the Classification Module
        Args:
            feature_map: torch.Tensor - input feature map
            proposals_list: List[torch.Tensor] - list of proposals for each image
            gt_classes: torch.Tensor - ground truth classes
        Returns:
            Union[torch.Tensor, torch.Tensor] - cls_scores, cls_loss
        Local variables shape:
            roi_out: torch.Tensor shape (N, C, roi_size, roi_size) - output from ROI pooling
            out: torch.Tensor shape (N, hidden_dim) - output from hidden network
            cls_scores: torch.Tensor shape (N, n_classes) - classification scores
            cls_loss: torch.Tensor - classification loss
        '''
        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        # apply roi pooling on proposals followed by avg pooling
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)
        
        # flatten the output
        roi_out = roi_out.squeeze(-1).squeeze(-1)
        
        # pass the output through the hidden network
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))
        
        # get the classification scores
        cls_scores = self.cls_head(out)
        
        if mode == 'eval':
            return cls_scores
        
        # compute cross entropy loss
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())
        
        return cls_loss
    
class TwoStageDetector(nn.Module):
    def __init__(self, 
                 img_size: Tuple[int, int], 
                 out_size: Tuple[int, int], 
                 out_channels: int, 
                 n_classes: int, 
                 roi_size: Tuple[int, int]):
        super().__init__() 
        self.rpn = RegionProposalNetwork(img_size, out_size, out_channels)
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size)
        
    def forward(self, 
                images: torch.Tensor, 
                gt_bboxes: torch.Tensor, 
                gt_classes: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the Two Stage Detector
        Args:
            images: torch.Tensor - input images
            gt_bboxes: torch.Tensor - ground truth bounding boxes
            gt_classes: torch.Tensor - ground truth classes
        Returns:
            torch.Tensor - total_loss
        Local variables shape:
            total_rpn_loss: torch.Tensor - total RPN loss
            feature_map: torch.Tensor shape (B, C, H, W) - output feature map from the backbone
            proposals: torch.Tensor shape (N, 4) - proposals generated by the RPN
            positive_anc_ind_sep: torch.Tensor shape (N, ) - indices of positive anchors separated by image
            GT_class_pos: torch.Tensor shape (N, ) - ground truth classes for positive anchors
            pos_proposals_list: List[torch.Tensor] - list of proposals for each image
            cls_loss: torch.Tensor - classification loss
            total_loss: torch.Tensor - total loss
        '''
        total_rpn_loss, feature_map, proposals, \
        positive_anc_ind_sep, GT_class_pos = self.rpn(images, gt_bboxes, gt_classes)
        
        # get separate proposals for each sample
        pos_proposals_list = []
        batch_size = images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)
        
        cls_loss = self.classifier(feature_map, pos_proposals_list, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss
        
        return total_loss
    
    def inference(self, 
                  images: torch.Tensor, 
                  conf_thresh: float, 
                  nms_thresh: float) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        '''
        Inference for the Two Stage Detector
        Args:
            images: torch.Tensor - input images
            conf_thresh: float - confidence threshold
            nms_thresh: float - NMS threshold
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]] - proposals_final, conf_scores_final, classes_final, probabs_final, embeddings_final
        Local variables shape:
            batch_size: int - number of samples in the batch
            proposals_final: List[torch.Tensor] - final proposals for each image
            conf_scores_final: List[torch.Tensor] - confidence scores for each proposal
            feature_map: torch.Tensor shape (B, C, H, W) - output feature map from the backbone
            roi_out: torch.Tensor shape (N, C, roi_size, roi_size) - output from ROI pooling
            out: torch.Tensor shape (N, hidden_dim) - output from hidden network
            cls_scores: torch.Tensor shape (N, n_classes) - classification scores
            cls_probs: torch.Tensor shape (N, n_classes) - class probabilities
            classes_all: torch.Tensor shape (N, ) - predicted classes
            classes_final: List[torch.Tensor] - predicted classes for each image
            probabs_final: List[torch.Tensor] - class probabilities for each image
            embeddings_final: List[torch.Tensor] - embeddings for each image
        '''
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)

        roi_out = ops.roi_pool(feature_map, proposals_final, self.classifier.roi_size)
        roi_out = self.classifier.avg_pool(roi_out)
        roi_out = roi_out.squeeze(-1).squeeze(-1)
        out = self.classifier.fc(roi_out)
        out = F.relu(self.classifier.dropout(out))
        cls_scores = self.classifier.cls_head(out)
        cls_probs = F.softmax(cls_scores, dim=-1)
        classes_all = torch.argmax(cls_probs, dim=-1)
        
        classes_final = []
        probabs_final = []
        embeddings_final = []
        c = 0
        for i in range(batch_size):
            n_proposals = len(proposals_final[i])
            classes_final.append(classes_all[c: c+n_proposals])
            probabs_final.append(cls_probs[c: c+n_proposals])
            embeddings_final.append(out[c: c+n_proposals])
            c += n_proposals
            
        return proposals_final, conf_scores_final, classes_final, probabs_final, embeddings_final

# ------------------- Loss Utils ----------------------

def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_scores_neg)
    
    target = torch.cat((target_pos, target_neg))
    inputs = torch.cat((conf_scores_pos, conf_scores_neg))
     
    loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='sum') * 1. / batch_size
    
    return loss

def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
    assert gt_offsets.size() == reg_offsets_pos.size()
    loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') * 1. / batch_size
    return loss