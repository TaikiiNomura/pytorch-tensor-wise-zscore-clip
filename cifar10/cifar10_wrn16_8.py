import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tpzc.adagc import AdaGC
from tpzc.zclip import ZClip
from tpzc.tpzc import TPZC


class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.dropout_rate = dropout_rate

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)

        shortcut = self.shortcut(out)

        out = self.conv1(out)

        if self.dropout_rate > 0.0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = out + shortcut
        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        depth=16,
        widen_factor=4,
        num_classes=10,
        dropout_rate=0.0,
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, "depth must be 6n + 4"
        n = (depth - 4) // 6

        channels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        self.conv1 = nn.Conv2d(
            3,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.layer1 = self._make_layer(
            channels[0],
            channels[1],
            num_blocks=n,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.layer2 = self._make_layer(
            channels[1],
            channels[2],
            num_blocks=n,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.layer3 = self._make_layer(
            channels[2],
            channels[3],
            num_blocks=n,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        in_channels,
        out_channels,
        num_blocks,
        stride,
        dropout_rate,
    ):
        layers = []

        layers.append(
            WideBasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout_rate=dropout_rate,
            )
        )

        for _ in range(1, num_blocks):
            layers.append(
                WideBasicBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    dropout_rate=dropout_rate,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.bn(out)
        out = F.relu(out)

        out = F.avg_pool2d(out, kernel_size=8)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out


def wrn16_4(num_classes=10, dropout_rate=0.0):
    return WideResNet(
        depth=16,
        widen_factor=4,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def wrn16_8(num_classes=10, dropout_rate=0.0):
    return WideResNet(
        depth=16,
        widen_factor=8,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


def get_datasets(
        data_dir=None,
):
    if data_dir is None:
        raise ValueError("data_dir must be specified")

    DATA_DIR = data_dir
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


def get_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=128,
    num_workers=2,
):


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


@torch.no_grad()
def eval_model(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        clip_info=None,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, t in train_loader:
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, t)
        loss.backward()

        if clip_info is not None:
            clip_type = clip_info["type"]

            if clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_info["clip_value"],
                )

            elif clip_type in ["zclip", "adagc", "tpzc"]:
                clipper = clip_info["clipper"]
                clipper.apply(model.parameters())

            else:
                raise ValueError(f"Unknown clip type: {clip_type}")

        optimizer.step()

        batch_size = t.size(0)
        total_loss += loss.item() * batch_size

        preds = output.argmax(dim=1)
        total_correct += (preds == t).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def set_device():
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if not device:
        raise ValueError("device not found")

    return device

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1111)

    parser.add_argument("--lr", type=float, default=1.5)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--data_dir", type=str, default="/home/user/workspace/data")
    parser.add_argument("--save_dir", type=str, default=".")

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model", type=str, default="wrn16_8",
                        choices=["wrn16_4", "wrn16_8"])
    parser.add_argument("--dropout_rate", type=float, default=0.0)

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.97)
    parser.add_argument("--c_z", type=float, default=3.0)
    parser.add_argument("--warmup_steps", type=int, default=25)

    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--eta_min_ratio", type=float, default=0.01)

    args = parser.parse_args()

    device = set_device()
    print(device)

    print(
        f"{device} | seed:{args.seed} | lr:{args.lr} | bs:{args.bs} | "
        f"beta:{args.beta} | c_z:{args.c_z} | warmup:{args.warmup_steps} | "
        f"scheduler:{args.use_scheduler}"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    train_datasets, test_datasets = get_datasets(args.data_dir)

    set_seed(args.seed)

    if args.model == "wrn16_4":
        init_model = wrn16_4(
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate,
        )
    elif args.model == "wrn16_8":
        init_model = wrn16_8(
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    torch.save(init_model.state_dict(), "init_model.pt")

    optimizer_and_clipper = {
        "SGD": {
            "func": lambda params: torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
            "clip_info": None,
        },

        "NormClip": {
            "func": lambda params: torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
            "clip_info": {
                "type": "norm",
                "clip_value": args.max_norm,
            },
        },

        "ZClip": {
            "func": lambda params: torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
            "clip_info": {
                "type": "zclip",
                "clipper": ZClip(
                    beta=args.beta,
                    c_z=args.c_z,
                    warmup_steps=args.warmup_steps,
                ),
            },
        },

        "AdaGC": {
            "func": lambda params: torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
            "clip_info": {
                "type": "adagc",
                "clipper": AdaGC(
                    beta=args.beta,
                    warmup_steps=args.warmup_steps,
                ),
            },
        },

        "TPZC": {
            "func": lambda params: torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
            "clip_info": {
                "type": "tpzc",
                "clipper": TPZC(
                    beta=args.beta,
                    c_z=args.c_z,
                    warmup_steps=args.warmup_steps,
                ),
            },
        },
    }

    criterion = torch.nn.CrossEntropyLoss()

    for optimizer_name, optimizer_info in optimizer_and_clipper.items():
        print(f"optimizer: {optimizer_name}")

        set_seed(args.seed)

        if args.model == "wrn16_4":
            model = wrn16_4(
                num_classes=args.num_classes,
                dropout_rate=args.dropout_rate,
            )
        elif args.model == "wrn16_8":
            model = wrn16_8(
                num_classes=args.num_classes,
                dropout_rate=args.dropout_rate,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        model.load_state_dict(
            torch.load("init_model.pt", map_location="cpu")
        )

        model.to(device)

        train_loader, test_loader = get_dataloaders(
            train_datasets,
            test_datasets,
            args.bs,
            args.workers,
        )

        optimizer = optimizer_info["func"](model.parameters())
        clip_info = optimizer_info["clip_info"]

        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * args.eta_min_ratio,
            )
        else:
            scheduler = None

        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        lr_hist = []

        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                clip_info,
            )

            test_loss, test_acc = eval_model(
                model,
                test_loader,
                criterion,
                device,
            )

            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            lr_hist.append(current_lr)

            print(
                "epoch:{:02d} | lr:{:.6f} | train loss:{:.6f} | train acc:{:.6f} | "
                "test loss:{:.6f} | test acc:{:.6f}".format(
                    epoch,
                    current_lr,
                    train_loss,
                    train_acc,
                    test_loss,
                    test_acc,
                )
            )

            if scheduler is not None:
                scheduler.step()

        scheduler_name = "cosine" if args.use_scheduler else "none"

        save_name = (
            f"cifar10_{args.model}_{optimizer_name}"
            f"_lr{args.lr}_bs{args.bs}_seed{args.seed}"
            f"_cz{args.c_z}_beta{args.beta}_sched{scheduler_name}.csv"
        )

        save_path = os.path.join(args.save_dir, save_name)

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"]
            )

            for i in range(args.epochs):
                writer.writerow(
                    [
                        i,
                        lr_hist[i],
                        train_loss_hist[i],
                        train_acc_hist[i],
                        test_loss_hist[i],
                        test_acc_hist[i],
                    ]
                )

        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()