"""训练 TinyCNN 并导出 artifacts/tiny_cnn.pt + model.onnx。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from sut.tiny_model import SEED, TinyCNN, make_synthetic_dataset

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    x, y = make_synthetic_dataset()
    split = int(x.shape[0] * 0.8)
    train_x, train_y, val_x, val_y = x[:split], y[:split], x[split:], y[split:]

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, 9):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(val_x).argmax(1) == val_y).float().mean().item()
        print(f"epoch {epoch}: loss={loss.item():.4f} val_acc={acc:.4f}")

    torch.save(model.state_dict(), ARTIFACTS / "tiny_cnn.pt")
    batch_dim = torch.export.Dim("batch")
    torch.onnx.export(model, train_x[:1], str(ARTIFACTS / "model.onnx"),
                      input_names=["input"], output_names=["logits"],
                      dynamic_shapes={"x": {0: batch_dim}})
    print("saved:", ARTIFACTS / "tiny_cnn.pt")
    print("saved:", ARTIFACTS / "model.onnx")


if __name__ == "__main__":
    main()