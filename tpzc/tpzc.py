import math
import torch


class TensorStat:
    def __init__(self, x):
        self.mean = x
        self.var = 0.0
        self.count = 1


class TPZC:
    def __init__(
        self,
        beta=0.97,
        c_z=2.5,
        warmup_steps=25,
        eps=1e-8,
    ):
        self.beta = beta
        self.c_z = c_z
        self.warmup_steps = warmup_steps
        self.eps = eps

        self.step = 0
        self.state = {}

    def _calc_z_score(self, h, mean, var):
        std = math.sqrt(var + self.eps)
        z = (h - mean) / (std + self.eps)
        return z, std

    def _update_stat(self, stat, x):
        new_mean = (
            self.beta * stat.mean
            + (1.0 - self.beta) * x
        )

        new_var = (
            self.beta * stat.var
            + (1.0 - self.beta) * ((x - new_mean) ** 2)
        )

        stat.mean = new_mean
        stat.var = new_var
        stat.count += 1

    @torch.no_grad()
    def apply(self, params):
        params = list(params)
        self.step += 1

        for p in params:
            if p.grad is None:
                continue

            g = p.grad
            key = id(p)

            # tensor-wise grad norm
            h = torch.norm(g.detach().float(), p=2).item()

            # 初回はclipせず統計を初期化
            if key not in self.state:
                self.state[key] = TensorStat(h)
                continue

            stat = self.state[key]

            h_star = h
            scale = 1.0

            # warmup後のみTensor-wise ZClip
            if self.step > self.warmup_steps:
                z, std = self._calc_z_score(
                    h=h,
                    mean=stat.mean,
                    var=stat.var,
                )

                if z > self.c_z:
                    # reciprocal clipping
                    z_star = (self.c_z ** 2) / z

                    # h* = mu + z* sigma
                    h_star = stat.mean + z_star * std

                    # tensor-wise scale
                    scale = min(
                        1.0,
                        h_star / (h + self.eps),
                    )

                    if scale < 1.0:
                        g.mul_(scale)

                    # 実際のclip後norm
                    h_star = h * scale

            # 統計更新：clip後ノルムで更新
            self._update_stat(stat, h_star)