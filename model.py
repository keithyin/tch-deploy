import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bigru = nn.GRU(
            10, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(in_features=2 * 128, out_features=5, bias=False)

    def forward(self, batch_feature):
        x = batch_feature
        x, _ = self.bigru(x)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.sum(dim=1)
        x = self.linear(x)
        x = torch.softmax(x, dim=1)
        return x


"""
    torch.onnx.export(
        model,
        (example, example_lengths),
        input_names=["feat", "len"],
        output_names=["prob"],
        dynamic_axes={
            "feat": {0: "batch", 1: "sequence"},
            "len": {0: "batch"},
            "prob": {0: "batch"},
        },
        f="model.onnx",
    )
"""

if __name__ == "__main__":
    gpu_device = torch.device("cuda")
    model = Model().to(gpu_device)
    example = torch.rand(2, 10, 10).to(gpu_device)

    # 跟踪模型
    traced_script_module = torch.jit.trace(model, [example])

    # 保存模型
    traced_script_module.save("traced_model.pt")
    torch.onnx.export(
        model,
        (example),
        input_names=["feat"],
        output_names=["prob"],
        dynamic_axes={
            "feat": {0: "batch", 1: "sequence"},
            "prob": {0: "batch"},
        },
        f="model.onnx",
    )
