from pyutils.loss.kd import kd_loss, KDLoss
from pyutils.loss.dkd import dkd_loss, DKDLoss

import torch


def test_kd():
    y = torch.randn(2, 10, requires_grad=True)
    teacher_scores = torch.randn(2, 10, requires_grad=True)
    labels = torch.randint(0, 10, (2,))
    T = 6.0
    alpha = 0.9
    loss = kd_loss(y, teacher_scores, labels, T, alpha, logit_stand=True)
    print(loss)
    loss.backward()


def test_dkd():
    y = torch.randn(2, 10, requires_grad=True)
    teacher_scores = torch.randn(2, 10, requires_grad=True)
    labels = torch.randint(0, 10, (2,))
    T = 6.0
    alpha = 0.9
    beta = 1
    loss = dkd_loss(y, teacher_scores, labels, alpha, beta, T, logit_stand=True)
    print(loss)
    loss.backward()


if __name__ == "__main__":
    test_kd()
    test_dkd()
