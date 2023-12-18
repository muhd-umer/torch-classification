import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            base_optimizer (torch.optim.Optimizer): base optimizer to use (e.g., SGD, Adam)
            rho (float, optional): neighborhood size. Defaults to 0.05.
            adaptive (bool, optional): whether to use adaptive neighborhood size. Defaults to False.
            **kwargs: additional arguments for the base optimizer
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Perform the first step of SAM, which computes the gradient and moves the parameters in the direction of the gradient.

        Args:
            zero_grad (bool, optional): whether to zero the gradients after the step. Defaults to False.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Perform the second step of SAM, which moves the parameters back to their original values and performs the update of the base optimizer.

        Args:
            zero_grad (bool, optional): whether to zero the gradients after the step. Defaults to False.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a step of SAM, which includes both the first and second steps.

        Args:
            closure (callable, optional): A closure that reevaluates the model and computes the loss. Defaults to None.
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        Compute the norm of the gradients.

        Returns:
            torch.Tensor: the norm of the gradients
        """
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        """
        Load state dictionary.

        Args:
            state_dict (dict): state dictionary to load
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
