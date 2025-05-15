
import torch
import torch.nn as nn
from .config import DefaultConfig




def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords

class GenTargets(nn.Module):
    def __init__(self,strides,limit_range):
        super().__init__()
        self.strides=strides
        self.limit_range=limit_range
        assert len(strides)==len(limit_range)

    def forward(self,inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits,cnt_logits,reg_preds=inputs[0]
        gt_boxes=inputs[1]
        classes=inputs[2]
        cls_targets_all_level=[]
        cnt_targets_all_level=[]
        reg_targets_all_level=[]
        assert len(self.strides)==len(cls_logits)
        for level in range(len(cls_logits)):
            level_out=[cls_logits[level],cnt_logits[level],reg_preds[level]]
            level_targets=self._gen_level_targets(level_out,gt_boxes,classes,self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
            
        return torch.cat(cls_targets_all_level,dim=1),torch.cat(cnt_targets_all_level,dim=1),torch.cat(reg_targets_all_level,dim=1)

    def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits,cnt_logits,reg_preds=out
        batch_size=cls_logits.shape[0]
        class_num=cls_logits.shape[1]
        m=gt_boxes.shape[1]

        cls_logits=cls_logits.permute(0,2,3,1) #[batch_size,h,w,class_num]  
        coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2]

        cls_logits=cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]  
        cnt_logits=cnt_logits.permute(0,2,3,1)
        cnt_logits=cnt_logits.reshape((batch_size,-1,1))
        reg_preds=reg_preds.permute(0,2,3,1)
        reg_preds=reg_preds.reshape((batch_size,-1,4))

        h_mul_w=cls_logits.shape[1]

        x=coords[:,0]
        y=coords[:,1]
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,:]
        r_off=gt_boxes[...,2][:,None,:]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,:]-y[None,:,None]
        ltrb_off=torch.stack([l_off,t_off,r_off,b_off],dim=-1)#[batch_size,h*w,m,4]

        areas=(ltrb_off[...,0]+ltrb_off[...,2])*(ltrb_off[...,1]+ltrb_off[...,3])#[batch_size,h*w,m]

        off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
        off_max=torch.max(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]

        mask_in_gtboxes=off_min>0
        mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])

        radiu=stride*sample_radiu_ratio
        gt_center_x=(gt_boxes[...,0]+gt_boxes[...,2])/2
        gt_center_y=(gt_boxes[...,1]+gt_boxes[...,3])/2
        c_l_off=x[None,:,None]-gt_center_x[:,None,:]#[1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off=y[None,:,None]-gt_center_y[:,None,:]
        c_r_off=gt_center_x[:,None,:]-x[None,:,None]
        c_b_off=gt_center_y[:,None,:]-y[None,:,None]
        c_ltrb_off=torch.stack([c_l_off,c_t_off,c_r_off,c_b_off],dim=-1)#[batch_size,h*w,m,4]
        c_off_max=torch.max(c_ltrb_off,dim=-1)[0]
        mask_center=c_off_max<radiu

        mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]

        areas[~mask_pos]=99999999
        areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]
        reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        reg_targets=torch.reshape(reg_targets,(batch_size,-1,4))#[batch_size,h*w,4]

        classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]#[batch_size,h*w,m]
        cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]

        assert reg_targets.shape==(batch_size,h_mul_w,4)
        assert cls_targets.shape==(batch_size,h_mul_w,1)
        assert cnt_targets.shape==(batch_size,h_mul_w,1)

        #process neg coords
        mask_pos_2=mask_pos.long().sum(dim=-1)#[batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2=mask_pos_2>=1
        assert mask_pos_2.shape==(batch_size,h_mul_w)
        cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]
        cnt_targets[~mask_pos_2]=-1
        reg_targets[~mask_pos_2]=-1
        
        return cls_targets,cnt_targets,reg_targets
        
import torch.nn.functional as F
def bce_loss(preds, targets, mask):
    """
    Binary Cross-Entropy Loss
    Args:
    preds: [batch_size, height, width] predicted probabilities (0-1)
    targets: [batch_size, height, width] ground truth (0 for non-text, 1 for text)
    mask: [batch_size, height, width] mask for valid pixels (0 for invalid, 1 for valid)
    """
    preds = preds.sigmoid()  # Apply sigmoid to get probabilities
    targets = targets.float()

    # Only compute loss for valid pixels (those with mask=1)
    loss = F.binary_cross_entropy(preds, targets, reduction='none')  # per-pixel BCE loss
    loss = loss * mask  # Apply mask to ignore invalid pixels
    return loss.sum() / mask.sum()  # Normalize by the number of valid pixels

def dice_loss(preds, targets, mask):
    """
    Dice Loss
    Args:
    preds: [batch_size, height, width] predicted probabilities (0-1)
    targets: [batch_size, height, width] ground truth (0 for non-text, 1 for text)
    mask: [batch_size, height, width] mask for valid pixels (0 for invalid, 1 for valid)
    """
    preds = preds.sigmoid()  # Apply sigmoid to get probabilities
    targets = targets.float()

    intersection = torch.sum(preds * targets * mask)  # Intersection of predicted and true regions
    union = torch.sum(preds * mask) + torch.sum(targets * mask)  # Union of predicted and true regions

    dice = 2 * intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    return 1 - dice  # Return Dice Loss (1 - Dice coefficient)
def compute_cls_loss(preds, targets, mask, mode='bce_dice'):
    '''
    Args
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size, sum(_h*_w), 1] ground truth labels (1 for text, 0 for non-text)
    mask: [batch_size, sum(_h*_w)] mask for valid pixels (0 for invalid, 1 for valid)
    mode: 'bce_dice' to compute combined loss (BCE + Dice)
    '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)  # Ensure mask has the right dimensions
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # Count of valid positions

    # Reshape predictions from each level
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)  # Change dimensions from [batch_size, _h, _w, class_num]
        pred = torch.reshape(pred, [batch_size, -1, class_num])  # Flatten spatial dimensions
        preds_reshape.append(pred)

    preds = torch.cat(preds_reshape, dim=1)  # Combine all levels to form a single tensor
    assert preds.shape[:2] == targets.shape[:2]  # Ensure same batch size and height/width

    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w), class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w), 1]

        # Convert sparse targets to one-hot format
        target_pos = (torch.arange(1, class_num+1, device=target_pos.device)[None, :] == target_pos).float()

        # Focal Loss for classification
        cls_loss = focal_loss_from_logits(pred_pos, target_pos).view(1)

        # BCE Loss for text vs non-text classification
        bce = bce_loss(pred_pos, target_pos, mask[batch_index])

        # Dice Loss for text area segmentation
        dice = dice_loss(pred_pos, target_pos, mask[batch_index])

        # Combine losses
        if mode == 'bce_dice':
            total_loss = cls_loss + bce + dice
        else:
            raise NotImplementedError("Unknown loss mode")

        loss.append(total_loss)

    return torch.cat(loss, dim=0) / num_pos  # Normalize by the number of valid positions
# def compute_cls_loss(preds,targets,mask):
#     '''
#     Args
#     preds: list contains five level pred [batch_size,class_num,_h,_w]
#     targets: [batch_size,sum(_h*_w),1]
#     mask: [batch_size,sum(_h*_w)]
#     '''
#     batch_size=targets.shape[0]
#     preds_reshape=[]
#     class_num=preds[0].shape[1]
#     mask=mask.unsqueeze(dim=-1)
#     # mask=targets>-1#[batch_size,sum(_h*_w),1]
#     num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
#     for pred in preds:
#         pred=pred.permute(0,2,3,1)
#         pred=torch.reshape(pred,[batch_size,-1,class_num])
#         preds_reshape.append(pred)
#     preds=torch.cat(preds_reshape,dim=1)#[batch_size,sum(_h*_w),class_num]
#     assert preds.shape[:2]==targets.shape[:2]
#     loss=[]
#     for batch_index in range(batch_size):
#         pred_pos=preds[batch_index]#[sum(_h*_w),class_num]
#         target_pos=targets[batch_index]#[sum(_h*_w),1]
#         target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
#         loss.append(focal_loss_from_logits(pred_pos,target_pos).view(1))
#     return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_cnt_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),1]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,]
        assert len(pred_pos.shape)==1
        loss.append(nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_reg_loss(preds,targets,mask,mode='giou'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos=torch.sum(mask,dim=1).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        assert len(pred_pos.shape)==2
        if mode=='iou':
            loss.append(iou_loss(pred_pos,target_pos).view(1))
        elif mode=='giou':
            loss.append(giou_loss(pred_pos,target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]


# import torch
#
#
# def direction_loss(preds, targets):
#     '''
#     Args:
#     preds: [n, 1] predicted angles
#     targets: [n, 1] ground truth angles
#     '''
#     # 计算角度差异，假设角度在[-pi, pi]之间
#     angle_diff = torch.abs(preds - targets)
#     angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)  # 处理周期性
#     loss = angle_diff
#     return loss.mean()
#
#
# def compute_reg_loss(preds, targets, mask, mode='giou', angle_preds=None, angle_targets=None, angle_weight=0.1):
#     '''
#     Args:
#     preds: list contains five level pred [batch_size, 4, _h, _w]
#     targets: [batch_size, sum(_h*_w), 4]
#     mask: [batch_size, sum(_h*_w)]
#     angle_preds: [batch_size, sum(_h*_w), 1] predicted angles
#     angle_targets: [batch_size, sum(_h*_w), 1] ground truth angles
#     angle_weight: a factor to control the contribution of direction loss
#     '''
#     batch_size = targets.shape[0]
#     c = targets.shape[-1]
#     preds_reshape = []
#     num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()
#
#     # Flatten preds for all levels
#     for pred in preds:
#         pred = pred.permute(0, 2, 3, 1)
#         pred = torch.reshape(pred, [batch_size, -1, c])
#         preds_reshape.append(pred)
#     preds = torch.cat(preds_reshape, dim=1)
#
#     assert preds.shape == targets.shape
#
#     loss = []
#     for batch_index in range(batch_size):
#         pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b, 4]
#         target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b, 4]
#         assert len(pred_pos.shape) == 2
#
#         if mode == 'iou':
#             loss.append(iou_loss(pred_pos, target_pos).view(1))
#         elif mode == 'giou':
#             loss.append(giou_loss(pred_pos, target_pos).view(1))
#         else:
#             raise NotImplementedError("reg loss only implemented ['iou', 'giou']")
#
#         # 如果提供了角度预测和目标
#         if angle_preds is not None and angle_targets is not None:
#             angle_diff_loss = direction_loss(angle_preds[batch_index][mask[batch_index]],
#                                              angle_targets[batch_index][mask[batch_index]])
#             loss[-1] += angle_weight * angle_diff_loss.view(1)
#
#     return torch.cat(loss, dim=0) / num_pos


def iou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt=torch.min(preds[:,:2],targets[:,:2])
    rb=torch.min(preds[:,2:],targets[:,2:])
    wh=(rb+lt).clamp(min=0)
    overlap=wh[:,0]*wh[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    iou=overlap/(area1+area2-overlap)
    loss=-iou.clamp(min=1e-6).log()
    return loss.sum()

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds=preds.sigmoid()
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*targets+(1.0-alpha)*(1.0-targets)
    loss=-w*torch.pow((1.0-pt),gamma)*pt.log()
    return loss.sum()




class LOSS(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds,targets=inputs
        cls_logits,cnt_logits,reg_preds=preds
        cls_targets,cnt_targets,reg_targets=targets
        mask_pos=(cnt_targets>-1).squeeze(dim=-1)# [batch_size,sum(_h*_w)]
        cls_loss=compute_cls_loss(cls_logits,cls_targets,mask_pos).mean()#[]
        cnt_loss=compute_cnt_loss(cnt_logits,cnt_targets,mask_pos).mean()
        reg_loss=compute_reg_loss(reg_preds,reg_targets,mask_pos).mean()
        if self.config.add_centerness:
            total_loss=cls_loss+cnt_loss+reg_loss
            return cls_loss,cnt_loss,reg_loss,total_loss
        else:
            total_loss=cls_loss+reg_loss+cnt_loss*0.0
            return cls_loss,cnt_loss,reg_loss,total_loss





if __name__=="__main__":
    loss=compute_cnt_loss([torch.ones([2,1,4,4])]*5,torch.ones([2,80,1]),torch.ones([2,80],dtype=torch.bool))
    print(loss)




        


        































