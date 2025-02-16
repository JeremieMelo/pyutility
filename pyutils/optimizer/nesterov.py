##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
# @brief  Nesterov's accelerated gradient method proposed by e-place.
#

import torch
from torch.optim.optimizer import Optimizer, required

__all__ = ["NesterovAcceleratedGradientOptimizer"]


class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    @brief Follow the Nesterov's implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    """

    def __init__(
        self,
        params,
        lr: float = required,
        constraint_fn=None,
        use_bb: bool = False,
    ):
        """
        @brief initialization
        @param params variable to optimize
        @param lr learning rate
        @param obj_and_grad_fn a callable function to get objective and gradient
        @param constraint_fn a callable function to force variables to satisfy all the constraints
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution
        # v_k is reference solution
        # obj_k is the objective at v_k
        # a_k is optimization parameter
        # alpha_k is the step size
        # v_k_1 is previous reference solution
        # g_k_1 is gradient to v_k_1
        # obj_k_1 is the objective at v_k_1
        defaults = dict(
            lr=lr,
            u_k=[],
            v_k=[],
            g_k=[],
            obj_k=[],
            a_k=[],
            alpha_k=[],
            v_k_1=[],
            g_k_1=[],
            obj_k_1=[],
            v_kp1=[None],
            obj_eval_count=0,
        )
        super(NesterovAcceleratedGradientOptimizer, self).__init__(params, defaults)
        if constraint_fn is None:

            def dummy_constraint_fn(*args, **kwargs):
                pass

            constraint_fn = dummy_constraint_fn
        self.constraint_fn = constraint_fn
        self.use_bb = use_bb

        # I do not know how to get generator's length
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

        self._params = self.param_groups[0]["params"]

    def __setstate__(self, state):
        super(NesterovAcceleratedGradientOptimizer, self).__setstate__(state)

    def step(self, closure):
        assert len(self.param_groups) == 1
        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        if self.use_bb:
            self.step_bb(closure)
        else:
            self.step_nobb(closure)

    def obj_and_grad_fn(self, p, v_k, closure):
        ## first load v_k to p
        p.grad = None
        old_p = p.data.clone()
        p.data.copy_(v_k.data)
        obj = closure()
        grad = p.grad
        p.grad = None
        p.data.copy_(old_p)
        return obj, grad

    def step_nobb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None

        obj_and_grad_fn = self.obj_and_grad_fn

        for group in self.param_groups:
            # obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group["params"]):
                if not group["u_k"]:
                    group["u_k"].append(p.data.clone())
                    group["v_k"].append(p)
                    obj, grad = obj_and_grad_fn(p, group["v_k"][i], closure)

                    group["g_k"].append(grad.data.clone())  # must clone
                    group["obj_k"].append(obj.data.clone())
                u_k = group["u_k"][i]
                v_k = group["v_k"][i]
                g_k = group["g_k"][i]
                obj_k = group["obj_k"][i]
                if not group["a_k"]:
                    group["a_k"].append(
                        torch.ones(1, dtype=g_k.dtype, device=g_k.device)
                    )
                    group["v_k_1"].append(
                        torch.autograd.Variable(
                            torch.zeros_like(v_k), requires_grad=True
                        )
                    )
                    group["v_k_1"][i].data.copy_(group["v_k"][i] - group["lr"] * g_k)
                    obj, grad = obj_and_grad_fn(p, group["v_k_1"][i], closure)
                    group["g_k_1"].append(grad.data)
                    group["obj_k_1"].append(obj.data.clone())
                a_k = group["a_k"][i]
                v_k_1 = group["v_k_1"][i]
                g_k_1 = group["g_k_1"][i]
                obj_k_1 = group["obj_k_1"][i]
                if not group["alpha_k"]:
                    group["alpha_k"].append(
                        (v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)
                    )
                alpha_k = group["alpha_k"][i]

                if group["v_kp1"][i] is None:
                    group["v_kp1"][i] = torch.autograd.Variable(
                        torch.zeros_like(v_k), requires_grad=True
                    )
                v_kp1 = group["v_kp1"][i]

                # line search with alpha_k as hint
                a_kp1 = a_k.square().add_(0.25).sqrt_().add_(0.5)
                coef = (a_k - 1) / a_kp1
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10

                while True:
                    u_kp1 = v_k - alpha_k * g_k
                    # constraint_fn(u_kp1)
                    v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                    # make sure v_kp1 subjects to constraints
                    # g_kp1 must correspond to v_kp1
                    constraint_fn(v_kp1)

                    f_kp1, g_kp1 = obj_and_grad_fn(p, v_kp1, closure)

                    # tt = time.time()
                    # alpha_kp1 = torch.sqrt(
                    #     torch.sum((v_kp1.data - v_k.data) ** 2)
                    #     / torch.sum((g_kp1.data - g_k.data) ** 2)
                    # )
                    alpha_kp1 = (
                        (v_kp1.data - v_k.data).square_().sum()
                        / (g_kp1.data - g_k.data).square_().sum()
                    ).sqrt_()
                    backtrack_cnt += 1
                    group["obj_eval_count"] += 1

                    if alpha_kp1 > 0.95 * alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        alpha_k.data.copy_(alpha_kp1.data)
                        break
                    else:
                        alpha_k.data.copy_(alpha_kp1.data)

                v_k_1.data.copy_(v_k.data)
                g_k_1.data.copy_(g_k.data)
                obj_k_1.data.copy_(obj_k.data)

                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)

        return loss

    def step_bb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group["params"]):
                if not group["u_k"]:
                    group["u_k"].append(p.data.clone())
                    group["v_k"].append(p)
                u_k = group["u_k"][i]
                v_k = group["v_k"][i]
                obj_k, g_k = obj_and_grad_fn(p, v_k, closure)
                if not group["obj_k"]:
                    group["obj_k"].append(None)
                group["obj_k"][i] = obj_k.data.clone()
                if not group["a_k"]:
                    group["a_k"].append(
                        torch.ones(1, dtype=g_k.dtype, device=g_k.device)
                    )
                    group["v_k_1"].append(
                        torch.autograd.Variable(
                            torch.zeros_like(v_k), requires_grad=True
                        )
                    )
                    group["v_k_1"][i].data.copy_(group["v_k"][i] - group["lr"] * g_k)
                a_k = group["a_k"][i]
                v_k_1 = group["v_k_1"][i]
                obj_k_1, g_k_1 = obj_and_grad_fn(p, v_k_1, closure)
                if not group["obj_k_1"]:
                    group["obj_k_1"].append(None)
                group["obj_k_1"][i] = obj_k_1.data.clone()
                if group["v_kp1"][i] is None:
                    group["v_kp1"][i] = torch.autograd.Variable(
                        torch.zeros_like(v_k), requires_grad=True
                    )
                v_kp1 = group["v_kp1"][i]
                if not group["alpha_k"]:
                    group["alpha_k"].append(
                        (v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)
                    )
                alpha_k = group["alpha_k"][i]
                # line search with alpha_k as hint
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1
                with torch.no_grad():
                    s_k = v_k - v_k_1
                    y_k = g_k - g_k_1
                    bb_short_step_size = (
                        s_k.flatten().dot(y_k.flatten())
                        / y_k.flatten().dot(y_k.flatten())
                    ).data
                    lip_step_size = (s_k.norm(p=2) / y_k.norm(p=2)).data
                    step_size = (
                        bb_short_step_size
                        if bb_short_step_size > 0
                        else min(lip_step_size, alpha_k)
                    )

                # one step
                u_kp1 = v_k - step_size * g_k
                v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                constraint_fn(v_kp1)
                group["obj_eval_count"] += 1

                v_k_1.data.copy_(v_k.data)
                alpha_k.data.copy_(step_size.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                a_k.data.copy_(a_kp1.data)

        return loss
