import torch

from pyutils.loss.dkd import DKDLoss, dkd_loss
from pyutils.loss.kd import KDLoss, kd_loss


def test_kd():
    y = torch.randn(2, 10, requires_grad=True)
    teacher_scores = torch.randn(2, 10, requires_grad=True)
    labels = torch.randint(0, 10, (2,))
    T = 2.0
    loss = kd_loss(
        y, teacher_scores, labels, T, ce_weight=0.1, kd_weight=9, logit_stand=True
    )
    print(loss)
    loss.backward()


def test_dkd():
    y = torch.randn(2, 10, requires_grad=True)
    teacher_scores = torch.randn(2, 10, requires_grad=True)
    labels = torch.randint(0, 10, (2,))
    T = 2.0
    alpha = 1
    beta = 1
    loss = dkd_loss(
        y,
        teacher_scores,
        labels,
        ce_weight=0.1,
        kd_alpha=alpha,
        kd_beta=beta,
        temperature=T,
        logit_stand=True,
    )
    print(loss)
    loss.backward()


if __name__ == "__main__":
    test_kd()
    test_dkd()
