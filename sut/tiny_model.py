"""TinyCNN：合成 5 类高斯数据集上的小型卷积网络（被测模型）。

刻意使用合成数据（离线可复现、CI 无外网依赖），
训练脚本 scripts/train_model.py 产出 artifacts/tiny_cnn.pt。
"""
import torch
from torch import nn

NUM_CLASSES = 5
IMG_SIZE = 8
SEED = 42


class TinyCNN(nn.Module):
    """两层卷积 + 两层全分类头，输入 (N,1,8,8)。"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # -> (8,8,8)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # -> (16,4,4)
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_synthetic_dataset(n_per_class: int = 200, seed: int = SEED):
    """生成 5 类高斯聚类图像数据 (N,1,8,8) 与标签。"""
    rng = torch.Generator().manual_seed(seed)
    centers = torch.randn(NUM_CLASSES, IMG_SIZE * IMG_SIZE, generator=rng) * 0.8
    xs, ys = [], []
    for cls in range(NUM_CLASSES):
        noise = torch.randn(n_per_class, IMG_SIZE * IMG_SIZE, generator=rng) * 0.5
        xs.append(centers[cls] + noise)
        ys.append(torch.full((n_per_class,), cls))
    x = torch.cat(xs).unsqueeze(1).view(-1, 1, IMG_SIZE, IMG_SIZE)
    y = torch.cat(ys)
    perm = torch.randperm(x.shape[0], generator=rng)
    return x[perm], y[perm]


def load_model(path="artifacts/tiny_cnn.pt") -> TinyCNN:
    """加载已训练 checkpoint（eval 态）。"""
    model = TinyCNN()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return model
