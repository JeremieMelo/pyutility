from pyutils.loss.kl_mixed import kl_mixed_loss, KLLossMixed
from pyutils.loss.dkd import dkd_loss, DKDLoss

import torch


def test_kl_mixed():
    y = torch.randn(2, 10, requires_grad=True)
    teacher_scores = torch.randn(2, 10, requires_grad=True)
    labels = torch.randint(0, 10, (2,))
    T = 6.0
    alpha = 0.9
    loss = kl_mixed_loss(y, teacher_scores, labels, T, alpha, logit_stand=True)
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
    test_kl_mixed()
    test_dkd()
