"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 02:10:59
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 02:10:59
"""

import torch
import numpy as np
from pyutils.quantize import *
from pyutils.general import TimerCtx

import matplotlib.pyplot as plt


def test_input_quant():
    bit = 8
    device = torch.device("cuda")
    qfn = input_quantize_fn(bit, alg="normal", device=device)
    qfn.train()
    for _ in range(10):
        x = torch.randn(640, device=device, requires_grad=True).clamp(0)
        x_q = qfn(x)

        print(x[:4])
        print(x_q[:4])
        print(qfn.scale, qfn.zero_point)
    print("test")
    qfn.eval()
    for _ in range(4):
        x = torch.randn(640, device=device).clamp_(0)
        x_q = qfn(x)

        print(x[:4])
        print(x_q[:4])
        print(qfn.scale, qfn.zero_point)


if __name__ == "__main__":
    test_input_quant()
