"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 03:14:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 03:14:43
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "KLLossMixed",
    "CrossEntropyLossSmooth",
    "f_divergence",
    "AdaptiveLossSoft",
    "FocalLossWithSmoothing",
    "DualFocalLoss",
]


class KLLossMixed(torch.nn.modules.loss._Loss):
    """
    description: Knowledge distillation loss function combines hard target loss and soft target loss, [https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L10]
    y {tensor.Tensor} Model output logits from the student model
    teacher_scores {tensor.Tensor} Model output logits from the teacher model
    labels {tensor.LongTensor} Hard labels, Ground truth
    T {scalar, Optional} temperature of the softmax function to make the teacher score softer. Default is 6 when accuracy is high, e.g., >95%. Typical value 1~20 [https://zhuanlan.zhihu.com/p/83456418]
    alpha {scalar, Optional} interpolation between hard and soft target loss. Default set to 0.9 for distillation mode. When hard target loss is very small (alpha=0.9), gets the best results. [https://zhuanlan.zhihu.com/p/102038521]
    return loss {tensor.Tensor} loss function
    """

    def __init__(self, T: float = 6.0, alpha: float = 0.9) -> None:
        """
        Args:
            T (float, optional): Temperature for softmax. Defaults to 6.0.
            alpha (float, optional): weighting factor for soft loss from teacher. Defaults to 0.9.
        """
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(
        self,
        y: Tensor,
        teacher_scores: Tensor,
        labels: Tensor,
    ) -> Tensor:

        alpha = np.clip(self.alpha, 0, 1)
        if alpha == 0:
            l_ce = F.cross_entropy(y, labels)
            return l_ce
        elif 0 < alpha < 1:
            p = F.log_softmax(y / self.T, dim=1)
            q = F.softmax(teacher_scores / self.T, dim=1)
            l_kl = F.kl_div(p, q, reduction="sum") * (self.T ** 2) / y.shape[0]
            l_ce = F.cross_entropy(y, labels)
            return l_kl * alpha + l_ce * (1.0 - alpha)
        else:
            p = F.log_softmax(y / self.T, dim=1)
            q = F.softmax(teacher_scores / self.T, dim=1)
            l_kl = F.kl_div(p, q, reduction="sum") * (self.T ** 2) / y.shape[0]
            return l_kl


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot.mul_(1 - self.eps).add_(self.eps / n_class)
        output_log_prob = F.log_softmax(output, dim=1)
        target.unsqueeze_(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def f_divergence(
    q_logits: Tensor, p_logits: Tensor, alpha: float, iw_clip: float = 1e3
) -> Tuple[Tensor, Tensor]:
    """https://github.com/facebookresearch/AlphaNet/blob/master/loss_ops.py
    title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
    author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
    """
    assert isinstance(alpha, float)
    q_prob = F.softmax(q_logits.data, dim=1)
    p_prob = F.softmax(p_logits.data, dim=1)
    # gradient is only backpropagated here
    q_log_prob = F.log_softmax(q_logits, dim=1)

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio.clamp_(0, iw_clip).log_()
        f = -importance_ratio
        f_base = 0
        rho_f = importance_ratio.sub_(1)
    elif abs(alpha - 1.0) < 1e-3:
        # f = importance_ratio * importance_ratio.log()
        f = importance_ratio.log().mul_(importance_ratio)
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha).clamp_(0, iw_clip)
        f_base = 1.0 / alpha / (alpha - 1.0)
        # f = iw_alpha / alpha / (alpha - 1.0)
        f = iw_alpha.mul(f_base)
        # f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha.div(alpha).add_(f_base)

    # loss = torch.sum(q_prob * (f - f_base), dim=1)
    loss = f.sub_(f_base).mul_(q_prob).sum(dim=1)
    grad_loss = -q_prob.mul_(rho_f).mul(q_log_prob).sum(dim=1)
    return loss, grad_loss


class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    """https://github.com/facebookresearch/AlphaNet/blob/master/loss_ops.py
    title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
    author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
    """

    def __init__(self, alpha_min: float, alpha_max: float, iw_clip: float = 1e3) -> None:
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ) -> Tensor:
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        loss_left, grad_loss_left = f_divergence(output, target, alpha_min, iw_clip=self.iw_clip)
        loss_right, grad_loss_right = f_divergence(output, target, alpha_max, iw_clip=self.iw_clip)

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLossWithSmoothing(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        gamma: int = 2,
        lb_smooth: float = 0.1,
        size_average: bool = True,
        ignore_index: int = None,
        alpha: float = None,
    ):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = torch.nn.LogSoftmax(dim=-1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError("The number of classes must be 2 or higher")
        if self._gamma < 0:
            raise ValueError("Gamma must be 0 or higher")
        if self._alpha is not None:
            if self._alpha < 0 or self._alpha > 1:
                raise ValueError("Alpha must be 0 < alpha < 1")

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        if self._num_classes == 2:
            difficulty_level = self._estimate_difficulty_level_binary(logits, label)
        else:
            difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self._lb_smooth, self._lb_smooth / (
                self._num_classes - 1
            )
            lb_one_hot = (
                torch.empty_like(logits)
                .fill_(lb_neg)
                .scatter_(1, label.unsqueeze(1), lb_pos)
                .detach()
            )
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * torch.nn.functional.softmax(logits, -1)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level

    def _estimate_difficulty_level_binary(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * torch.nn.functional.softmax(logits, -1)
        # pred = logits.data.max(1)[1]
        # fn = (label == 1) & (pred == 0)
        if self._alpha is not None:
            neg = label == 1
            alpha = torch.where(neg, 1 + self._alpha, 1 - self._alpha)
            difficulty_level = alpha.unsqueeze(1) * torch.pow(1 - pt, self._gamma)
        else:
            difficulty_level = torch.pow(1 - pt, self._gamma)

        return difficulty_level


class DualFocalLoss(torch.nn.modules.loss._Loss):
    """https://arxiv.org/abs/2305.13665
    Dual Focal Loss for Calibration
    """
    def __init__(self, gamma=5, size_average=True):
        super(DualFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logp_k = torch.nn.functional.log_softmax(input, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = (
            torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1
        )  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
