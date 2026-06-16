import math
import torch


class AdaGC:
    def __init__(
        self,
        beta=0.97,
        warmup_steps=25,
        eps=1e-8,
    ):
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.eps = eps

        self.step = 0
        self.mean = {}

    @torch.no_grad()
    def apply(self, params):
        params = list(params)
        self.step += 1

        for p in params:
            if p.grad is None:
                continue

            g = p.grad
            key = id(p)

            h = torch.norm(g.detach().float(), p=2).item()

            # 初回はclipせず平均を初期化
            if key not in self.mean:
                self.mean[key] = h
                continue

            gamma = self.mean[key]

            h_star = h
            scale = 1.0

            # warmup後のみAdaGC
            if self.step > self.warmup_steps:
                scale = min(
                    1.0,
                    gamma / (h + self.eps),
                )

                if scale < 1.0:
                    g.mul_(scale)

                # 実際のclip後norm
                h_star = h * scale

            # 統計更新：clip後ノルムで更新
            self.mean[key] = (
                self.beta * gamma
                + (1.0 - self.beta) * h_star
            )

