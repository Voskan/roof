
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


def _weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element/sample weighting then reduce loss."""
    if weight is not None:
        if not torch.is_tensor(weight):
            weight = torch.tensor(weight, dtype=loss.dtype, device=loss.device)
        weight = weight.to(device=loss.device, dtype=loss.dtype)
        while weight.ndim < loss.ndim:
            weight = weight.unsqueeze(-1)
        loss = loss * weight

    if reduction == 'none':
        return loss

    if avg_factor is None:
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        raise ValueError(f'Unsupported reduction={reduction}')

    if torch.is_tensor(avg_factor):
        avg_factor = float(avg_factor.detach().item())
    avg_factor = max(float(avg_factor), 1e-12)

    if reduction == 'mean':
        return loss.sum() / avg_factor
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(f'Unsupported reduction={reduction}')


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-12)
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


@MODELS.register_module()
class DeepRoofLosses(nn.Module):
    """
    Wrapper for DeepRoof-2026 specific losses.
    Although MMSeg provides standard losses, we implement them here for custom flexibility
    and to satisfy the project requirements for distinct loss modules.
    
    This file contains:
    1. DiceLoss
    2. CrossEntropyLoss
    3. CosineSimilarityLoss (Geometry)
    """
    def __init__(self):
        super().__init__()

@MODELS.register_module(name='DeepRoofDiceLoss')
class DiceLoss(nn.Module):
    r"""
    Dice Loss for Segmentation.
    
    Formula:
    $$ L_{Dice} = 1 - \frac{2 \sum_{i} p_i g_i}{\sum_{i} p_i^2 + \sum_{i} g_i^2 + \epsilon} $$
    where $p_i$ is the prediction and $g_i$ is the ground truth.
    """
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, use_sigmoid=True, **kwargs):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """
        Args:
            pred (Tensor): Predicted logits/probabilities.
            target (Tensor): Ground truth masks (N, H, W)
        """
        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Invalid reduction_override={reduction_override}')

        if self.use_sigmoid:
            pred = pred.sigmoid()

        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1).float()

        intersection = (pred * target).sum(1)
        union = (pred**2).sum(1) + (target**2).sum(1)
        dice_score = (2. * intersection) / (union + self.eps)
        loss = 1. - dice_score

        loss = _weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss * self.loss_weight

@MODELS.register_module(name='DeepRoofCrossEntropyLoss')
class CrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss.
    """
    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        if class_weight is not None and not torch.is_tensor(class_weight):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.class_weight = class_weight
            
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """
        Args:
            cls_score (Tensor): The prediction.
            label (Tensor): The learning label of the prediction.
        """
        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Invalid reduction_override={reduction_override}')

        class_weight = None
        if self.class_weight is not None:
            class_weight = self.class_weight.to(device=cls_score.device, dtype=cls_score.dtype)

        if self.use_sigmoid:
            if label.shape != cls_score.shape:
                if label.ndim == cls_score.ndim - 1 and cls_score.size(-1) > 1:
                    label = F.one_hot(label.long(), num_classes=cls_score.size(-1)).to(dtype=cls_score.dtype)
                else:
                    label = label.reshape_as(cls_score).to(dtype=cls_score.dtype)
            else:
                label = label.to(dtype=cls_score.dtype)
            loss = F.binary_cross_entropy_with_logits(
                cls_score, label, reduction='none', pos_weight=class_weight)
            if loss.ndim > 1:
                loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        else:
            loss = F.cross_entropy(
                cls_score,
                label.long(),
                weight=class_weight,
                reduction='none',
                ignore_index=ignore_index)

        loss = _weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss * self.loss_weight


@MODELS.register_module(name='DeepRoofHybridMaskLoss')
class HybridMaskLoss(nn.Module):
    """
    Hybrid mask loss:
    L = w_bce * BCEWithLogits + w_lovasz * LovaszHinge
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Invalid reduction_override={reduction_override}')

        if target.shape != pred.shape:
            target = target.reshape_as(pred)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if bce.ndim > 1:
            bce = bce.mean(dim=tuple(range(1, bce.ndim)))

        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        lovasz_vals = []
        for i in range(pred_flat.shape[0]):
            lovasz_vals.append(_lovasz_hinge_flat(pred_flat[i], target_flat[i]))
        lovasz = torch.stack(lovasz_vals) if lovasz_vals else pred.sum() * 0.0

        loss = self.bce_weight * bce + self.lovasz_weight * lovasz
        loss = _weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss * self.loss_weight


@MODELS.register_module(name='DeepRoofBoundaryLoss')
class BoundaryLoss(nn.Module):
    """Boundary-aware BCE loss for dense binary masks."""

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    @staticmethod
    def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            return mask.reshape(mask.shape[0], -1)
        dil = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        boundary = (dil - ero).clamp(min=0.0, max=1.0)
        return boundary

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        reduction = reduction_override if reduction_override is not None else self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Invalid reduction_override={reduction_override}')

        if pred.shape != target.shape:
            target = target.reshape_as(pred)
        target = target.float()

        # If tensor is not image-like, fallback to BCE.
        if pred.ndim < 3:
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            if loss.ndim > 1:
                loss = loss.mean(dim=tuple(range(1, loss.ndim)))
            loss = _weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
            return loss * self.loss_weight

        pred_prob = torch.sigmoid(pred)
        pred_boundary = self._boundary_map(pred_prob)
        tgt_boundary = self._boundary_map(target)
        loss = F.binary_cross_entropy(pred_boundary, tgt_boundary, reduction='none')
        if loss.ndim > 1:
            loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        loss = _weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss * self.loss_weight


@MODELS.register_module(name='DeepRoofDiceBoundaryLoss')
class DiceBoundaryLoss(nn.Module):
    """Combine Dice and Boundary losses for mask quality."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        eps: float = 1e-6,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.dice = DiceLoss(
            eps=eps,
            reduction=reduction,
            loss_weight=1.0,
            use_sigmoid=use_sigmoid)
        self.boundary = BoundaryLoss(
            reduction=reduction,
            loss_weight=1.0)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight)
        self.loss_weight = float(loss_weight)

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        dice_loss = self.dice(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)
        boundary_loss = self.boundary(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)
        return (self.dice_weight * dice_loss + self.boundary_weight * boundary_loss) * self.loss_weight


@MODELS.register_module(name='DeepRoofDenseNormalLoss')
class DenseNormalLoss(nn.Module):
    """
    Dense normal supervision:
    - angular term: 1 - cos(pred, target)
    - robust L1 term on components
    """

    def __init__(
        self,
        angular_weight: float = 1.0,
        l1_weight: float = 0.5,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.angular_weight = float(angular_weight)
        self.l1_weight = float(l1_weight)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

    def forward(self, pred, target, valid_mask=None, **kwargs):
        # pred/target: [B, 3, H, W]
        pred_n = F.normalize(pred, p=2, dim=1, eps=self.eps)
        target_n = F.normalize(target, p=2, dim=1, eps=self.eps)
        cos = (pred_n * target_n).sum(dim=1).clamp(min=-1.0, max=1.0)
        angular = 1.0 - cos
        l1 = torch.abs(pred_n - target_n).mean(dim=1)
        loss = self.angular_weight * angular + self.l1_weight * l1

        if valid_mask is not None:
            vm = valid_mask.float()
            while vm.ndim < loss.ndim:
                vm = vm.unsqueeze(0)
            if vm.ndim == 3 and vm.shape[0] != loss.shape[0]:
                vm = vm.unsqueeze(0)
            if vm.ndim == 4:
                vm = vm[:, 0]
            loss = loss * vm
            denom = vm.sum().clamp(min=1.0)
            out = loss.sum() / denom
            return out * self.loss_weight

        if self.reduction == 'mean':
            return loss.mean() * self.loss_weight
        if self.reduction == 'sum':
            return loss.sum() * self.loss_weight
        return loss * self.loss_weight


@MODELS.register_module(name='DeepRoofSAMDistillLoss')
class SAMDistillLoss(nn.Module):
    """Distill roof mask logits to SAM teacher mask."""

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(self, pred_logits, teacher_mask, valid_mask=None, **kwargs):
        # pred_logits, teacher_mask: [B, H, W]
        teacher = teacher_mask.float()
        loss = F.binary_cross_entropy_with_logits(pred_logits, teacher, reduction='none')
        if valid_mask is not None:
            vm = valid_mask.float()
            while vm.ndim < loss.ndim:
                vm = vm.unsqueeze(0)
            loss = loss * vm
            denom = vm.sum().clamp(min=1.0)
            out = loss.sum() / denom
            return out * self.loss_weight
        if self.reduction == 'mean':
            return loss.mean() * self.loss_weight
        if self.reduction == 'sum':
            return loss.sum() * self.loss_weight
        return loss * self.loss_weight

@MODELS.register_module()
class PhysicallyWeightedNormalLoss(nn.Module):
    r"""
    Physically-Weighted Normal Loss for SOTA Surface Regression.
    
    This loss handles the "Flat Roof Singularity" where azimuth is undefined.
    It splits the cosine similarity into a Z-component loss (slope) 
    and an (X,Y)-component loss (azimuth), where the azimuth penalty 
    is scaled by the ground-truth slope.
    
    Mathematical Formulation:
    $$ L_{geo} = (1 - nz_{pred} \cdot nz_{gt}) + \lambda_{az} \cdot \sqrt{1 - nz_{gt}^2} \cdot (1 - \frac{nx_{p}nx_{g} + ny_{p}ny_{g}}{\sqrt{1-nz_p^2}\sqrt{1-nz_g^2}}) $$
    """
    def __init__(self, reduction='mean', loss_weight=1.0, azimuth_weight=1.0, eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.azimuth_weight = azimuth_weight
        self.eps = eps

    def forward(self, pred, target, **kwargs):
        # pred, target: (N, 3)
        pred = F.normalize(pred, p=2, dim=-1, eps=self.eps)
        target = F.normalize(target, p=2, dim=-1, eps=self.eps)

        # 1. Slope Loss (nz component)
        loss_nz = (1.0 - pred[:, 2] * target[:, 2])

        # 2. Azimuth Weighting Factor (sin(theta_gt))
        # Scaled by how "sloped" the roof is. For flat roofs (nz=1), weight=0.
        az_weight = torch.sqrt((1.0 - target[:, 2]**2).clamp(min=0))

        # 3. Directional Alignment (nx, ny components normalized)
        # We only compute this if there is a horizontal component
        p_xy = pred[:, :2]
        t_xy = target[:, :2]
        
        p_xy_norm = F.normalize(p_xy, p=2, dim=-1, eps=self.eps)
        t_xy_norm = F.normalize(t_xy, p=2, dim=-1, eps=self.eps)
        
        loss_xy = (1.0 - (p_xy_norm * t_xy_norm).sum(dim=-1))
        
        # Combined Loss
        loss = loss_nz + self.azimuth_weight * az_weight * loss_xy

        # Mask background
        valid_mask = (target.abs().sum(dim=-1) > self.eps).float()
        loss = loss * valid_mask
        
        if self.reduction == 'mean':
            return (loss.sum() / valid_mask.sum().clamp(min=1.0)) * self.loss_weight
        return loss.sum() * self.loss_weight

@MODELS.register_module()
class CosineSimilarityLoss(nn.Module):
    r"""
    Cosine Similarity Loss for Geometry Regression (Normal Vectors).
    
    Mathematical Derivation:
    The dot product of two unit vectors $\mathbf{u}$ and $\mathbf{v}$ is defined as:
    $$ \mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta) $$
    
    Since we enforce $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$ (L2 normalization), this simplifies to:
    $$ \mathbf{u} \cdot \mathbf{v} = \cos(\theta) $$
    
    We want to minimize the angle $\theta$ between the predicted normal $\mathbf{n}_{pred}$ 
    and the ground truth normal $\mathbf{n}_{gt}$. 
    Maximizing $\cos(\theta)$ (towards 1) minimizes $\theta$ (towards 0).
    
    Therefore, the loss function is defined as:
    $$ L_{cos} = 1 - \cos(\theta) = 1 - (\mathbf{n}_{pred} \cdot \mathbf{n}_{gt}) $$
    
    Range:
    - If aligned ($\theta = 0^\circ$): $L = 1 - 1 = 0$ (Perfect)
    - If orthogonal ($\theta = 90^\circ$): $L = 1 - 0 = 1$
    - If opposite ($\theta = 180^\circ$): $L = 1 - (-1) = 2$ (Worst case)
    """
    def __init__(self, reduction='mean', loss_weight=1.0, eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): Predicted normal vectors (N, 3).
            target (Tensor): Ground truth normal vectors (N, 3).
                             Background/invalid pixels have GT normal = [0, 0, 0].
        """
        # Ensure inputs are L2-normalized (axis = last dim = 1 for (N,3))
        pred_norm = F.normalize(pred, p=2, dim=-1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=-1, eps=self.eps)

        # Cosine similarity = dot product of unit vectors
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)  # (N,)

        # Loss: 1 - cosine_sim  (range [0, 2]; 0=perfect, 2=opposite)
        loss = 1.0 - cosine_sim  # (N,)

        # FIX: Exclude zero-vector GT normals (background / unassigned instances).
        # Without this mask, instances with GT=[0,0,0] contribute loss≈1.0
        # which is meaningless noise — the model "learns" to predict flatness
        # even for unmatched queries just to minimize this spurious signal.
        valid_mask = (target.abs().sum(dim=-1) > self.eps).float()  # (N,)
        loss = loss * valid_mask

        n_valid = valid_mask.sum().clamp(min=1.0)

        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else 'none': return per-element

        return loss * self.loss_weight
