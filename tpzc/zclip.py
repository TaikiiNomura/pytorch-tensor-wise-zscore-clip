import math
import torch


class ZClip:
    def __init__(
        self,
        c_z=2.5,
        beta=0.97,
        warmup_steps=25,
        eps=1e-8,
    ):
        self.c_z = c_z
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.eps = eps

        self.step = 0
        self.mean = None
        self.var = 0.0

    def _global_grad_norm(self, params):
        total_sq = 0.0

        for p in params:
            if p.grad is None:
                continue

            g = p.grad.detach().float()
            total_sq += torch.sum(g * g).item()

        return math.sqrt(total_sq + self.eps)

    def _update_stat(self, x):
        if self.mean is None:
            self.mean = x
            self.var = 0.0
        else:
            new_mean = (
                self.beta * self.mean
                + (1.0 - self.beta) * x
            )

            new_var = (
                self.beta * self.var
                + (1.0 - self.beta) * ((x - new_mean) ** 2)
            )

            self.mean = new_mean
            self.var = new_var

    @torch.no_grad()
    def apply(self, params):
        params = list(params)
        self.step += 1

        G = self._global_grad_norm(params)

        G_star = G
        scale = 1.0

        # 初回はclipせず統計を初期化
        if self.mean is None:
            self._update_stat(G_star)
            return

        # warmup後のみZClip
        if self.step > self.warmup_steps:
            std = math.sqrt(self.var + self.eps)

            z = (G - self.mean) / (std + self.eps)

            if z > self.c_z:
                # reciprocal clipping
                z_star = (self.c_z ** 2) / z

                # G* = mu + z* sigma
                G_star = self.mean + z_star * std

                # global scale
                scale = min(
                    1.0,
                    G_star / (G + self.eps),
                )

                for p in params:
                    if p.grad is None:
                        continue
                    p.grad.mul_(scale)

                # 実際のclip後norm
                G_star = G * scale

        # 統計更新：clip後ノルムで更新
        self._update_stat(G_star)