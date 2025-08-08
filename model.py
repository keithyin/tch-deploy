import torch
from torch import nn
from torch.nn import functional as F

def sequence_mask_for_1dcnn(feature: torch.Tensor, lengths: torch.Tensor):
    """_summary_

    Args:
        feature (torch.Tensor): 3D tensor. [b, in_channel, tt]
        lengths (torch.Tensor): 1D tensor
        tt_axis (int): _description_
        dtype (_type_, optional): _description_. Defaults to torch.float32.

    Returns:
        2D tensor: [b, tt]
    """
    assert len(lengths.shape) == 1, f"{lengths.shape}, not 1D"

    mask = torch.ones(
        [lengths.shape[0], feature.shape[2]], device=feature.device
    ).cumsum(dim=1) <= torch.unsqueeze(lengths, 1)
    return mask

def emb_and_reshape(feat, emb_matrix):
    enc = F.embedding(feat, emb_matrix)
    enc = torch.reshape(
        enc, shape=[enc.shape[0], enc.shape[1], enc.shape[2] * enc.shape[3]]
    )
    return enc

class EmbLayerDcDw(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        assert feat_size == 61
        self.feat_size = 368
        self.base_emb = nn.Embedding(
            num_embeddings=5, padding_idx=0, embedding_dim=8)
        self.dw_emb = nn.Embedding(
            num_embeddings=256, padding_idx=0, embedding_dim=8)
        self.strand_emb = nn.Embedding(
            num_embeddings=3, padding_idx=0, embedding_dim=2)

    def forward(self, feature: torch.Tensor):
        sbr_bases = feature[:, :, 0:20].int()
        sbr_strand = feature[:, :, 20:40].int()
        sbr_dw = feature[:, :, 40:60].int()
        smc_bases = feature[:, :, 60:61].int()
        sbr_bases_enc = emb_and_reshape(sbr_bases, self.base_emb.weight)
        sbr_dw_enc = emb_and_reshape(sbr_dw, self.dw_emb.weight)
        sbr_strand_enc = emb_and_reshape(sbr_strand, self.strand_emb.weight)
        smc_bases_enc = emb_and_reshape(smc_bases, self.base_emb.weight)

        # [batch, tt, feat_dim]
        x = torch.concat(
            [sbr_bases_enc, sbr_dw_enc, sbr_strand_enc, smc_bases_enc], dim=2
        )

        return x


class EmbLayerDcDwArCr(nn.Module):
    def __init__(self, feat_size, disable_cr=0):
        super().__init__()
        assert feat_size == 101
        self.feat_size = 688
        self.base_emb = nn.Embedding(
            num_embeddings=5, padding_idx=0, embedding_dim=8)
        self.dw_emb = nn.Embedding(
            num_embeddings=256, padding_idx=0, embedding_dim=8)
        self.ar_emb = nn.Embedding(
            num_embeddings=256, padding_idx=0, embedding_dim=8)
        self.cr_emb = nn.Embedding(
            num_embeddings=256, padding_idx=0, embedding_dim=8)
        self.strand_emb = nn.Embedding(
            num_embeddings=3, padding_idx=0, embedding_dim=2)

    def forward(self, feature: torch.Tensor):
        sbr_bases = feature[:, :, 0:20].int()
        sbr_strand = feature[:, :, 20:40].int()
        sbr_dw = feature[:, :, 40:60].int()
        sbr_ar = feature[:, :, 60:80].int()
        sbr_cr = feature[:, :, 80:100].int()

        smc_bases = feature[:, :, 100:101].int()

        sbr_bases_enc = emb_and_reshape(sbr_bases, self.base_emb.weight)
        sbr_dw_enc = emb_and_reshape(sbr_dw, self.dw_emb.weight)
        sbr_strand_enc = emb_and_reshape(sbr_strand, self.strand_emb.weight)
        sbr_ar_enc = emb_and_reshape(sbr_ar, self.ar_emb.weight)
        sbr_cr_enc = emb_and_reshape(sbr_cr, self.cr_emb.weight)

        smc_bases_enc = emb_and_reshape(smc_bases, self.base_emb.weight)
        # print(smc_bases_enc.shape)
        x = torch.concat(
            [
                sbr_bases_enc,
                sbr_dw_enc,
                sbr_strand_enc,
                sbr_ar_enc,
                sbr_cr_enc,
                smc_bases_enc,
            ],
            dim=2,
        )

        return x


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


class ConsensusCnnModel(nn.Module):
    def __init__(self,
                 feat_size,
                 cnn_bias=True,
                 residual=False,
                 num_classes=5):
        super().__init__()
        self.residual = residual
        if feat_size == 61:
            self.emb_layer = EmbLayerDcDw(feat_size=feat_size)
        elif feat_size == 101:
            self.emb_layer = EmbLayerDcDwArCr(feat_size=feat_size)

        self.affine_layer = nn.Linear(
            self.emb_layer.feat_size, out_features=512, bias=False
        )

        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    padding="same",
                    dilation=2**i,
                    bias=cnn_bias
                )
                for i in range(6)
            ]
        )
        self.cls_layer = nn.Linear(512, out_features=num_classes, bias=False)

    def forward(self, feature: torch.Tensor, length: torch.Tensor):
        feature = self.emb_layer(feature)
        # [b, tt, featsize]
        feature = F.relu(self.affine_layer(feature))

        # [b, featsize, tt]
        feature = feature.transpose(1, 2)

        # [b, 1, tt]
        mask = sequence_mask_for_1dcnn(
            feature, length).unsqueeze(dim=1)

        for conv_layer in self.conv_list:
            feature = feature * mask
            x = conv_layer(feature)
            x = F.relu(x)
            if self.residual:
                feature = feature + x
            else:
                feature = x

        # [b, tt, featsize]
        feature = feature.transpose(1, 2)
        logits = self.cls_layer(feature)
        return logits

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
