# src/MODEL.py with padding to 5120 for even pooling, and debugging prints
import torch
import torch.nn as nn
import torch.nn.functional as F
from CONFIG import EMBED_DIM, NUM_RHYTHMS, NUM_CLASSES, P_ABSENT_IDS

class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout1d(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        return x

class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ConvBnRelu1d(in_channels, out_channels, kernel_size, padding)
        self.conv2 = ConvBnRelu1d(out_channels, out_channels, kernel_size, padding)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)

class StackDecoder3p(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv1d(ic, skip_channels, kernel_size, padding) for ic in in_channels])
        self.aggregate = ConvBnRelu1d(skip_channels * len(in_channels), out_channels, kernel_size, padding)

    def forward(self, *features):
        aggregated = []
        for conv, feat in zip(self.conv_layers, features):
            aggregated.append(conv(feat))
        x = torch.cat(aggregated, dim=1)
        x = self.aggregate(x)
        return x

class UNet3p(nn.Module):
    def __init__(self, n_channels=4, embed_dim=EMBED_DIM):
        super().__init__()
        filters = [n_channels * (2 ** n) for n in range(5)]  # [4,8,16,32,64]
        filters_skip = filters[0]  # 4
        filters_decoder = filters_skip * 5  # 20
        in_total = 1 + embed_dim  # ECG + embed
        self.rhythm_embed = nn.Embedding(NUM_RHYTHMS, embed_dim)

        self.down1 = StackEncoder(in_total, filters[0])
        self.down2 = StackEncoder(filters[0], filters[1])
        self.down3 = StackEncoder(filters[1], filters[2])
        self.down4 = StackEncoder(filters[2], filters[3])
        self.middle = nn.Sequential(ConvBnRelu1d(filters[3], filters[4]), ConvBnRelu1d(filters[4], filters[4]))

        self.up4 = StackDecoder3p(filters, filters_skip, filters_decoder)
        self.up3 = StackDecoder3p(filters[:3] + [filters_decoder] + filters[4:], filters_skip, filters_decoder)
        self.up2 = StackDecoder3p(filters[:2] + [filters_decoder] * 2 + filters[4:], filters_skip, filters_decoder)
        self.up1 = StackDecoder3p(filters[:1] + [filters_decoder] * 3 + filters[4:], filters_skip, filters_decoder)
        self.segment = nn.Conv1d(filters_decoder, NUM_CLASSES, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_ecg, rid, suppress_p=False):
        B, _, L = x_ecg.shape
        print(f"Input length: {L}")
        # Pad to 5120 (next multiple of 32 for 5 levels)
        pad_total = 5120 - L
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_ecg = F.pad(x_ecg, (pad_left, pad_right), mode='replicate')
        print(f"Padded length: {x_ecg.shape[-1]}")
        embed = self.rhythm_embed(rid).unsqueeze(-1).expand(-1, -1, x_ecg.size(2))
        x = torch.cat([x_ecg, embed], dim=1)

        # Encoder with prints
        X_enc1, x = self.down1(x)
        print(f"X_enc1 length: {X_enc1.shape[-1]}")
        X_enc2, x = self.down2(x)
        print(f"X_enc2 length: {X_enc2.shape[-1]}")
        X_enc3, x = self.down3(x)
        print(f"X_enc3 length: {X_enc3.shape[-1]}")
        X_enc4, x = self.down4(x)
        print(f"X_enc4 length: {X_enc4.shape[-1]}")
        X_enc5 = self.middle(x)
        print(f"X_enc5 length: {X_enc5.shape[-1]}")

        # Decoder with prints
        X_dec5 = X_enc5
        X_dec4 = self.up4(
            F.max_pool1d(X_enc1, kernel_size=8, stride=8),
            F.max_pool1d(X_enc2, kernel_size=4, stride=4),
            F.max_pool1d(X_enc3, kernel_size=2, stride=2),
            X_enc4,
            F.interpolate(X_dec5, size=X_enc4.shape[-1], mode='linear', align_corners=False)
        )
        print(f"X_dec4 length: {X_dec4.shape[-1]}")
        X_dec3 = self.up3(
            F.max_pool1d(X_enc1, kernel_size=4, stride=4),
            F.max_pool1d(X_enc2, kernel_size=2, stride=2),
            X_enc3,
            F.interpolate(X_dec4, size=X_enc3.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc3.shape[-1], mode='linear', align_corners=False)
        )
        print(f"X_dec3 length: {X_dec3.shape[-1]}")
        X_dec2 = self.up2(
            F.max_pool1d(X_enc1, kernel_size=2, stride=2),
            X_enc2,
            F.interpolate(X_dec3, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc2.shape[-1], mode='linear', align_corners=False)
        )
        print(f"X_dec2 length: {X_dec2.shape[-1]}")
        X_dec1 = self.up1(
            X_enc1,
            F.interpolate(X_dec2, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec3, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc1.shape[-1], mode='linear', align_corners=False)
        )
        print(f"X_dec1 length: {X_dec1.shape[-1]}")
        seg_logits = self.segment(X_dec1)
        print(f"seg_logits length: {seg_logits.shape[-1]}")

        # Crop back to original L
        seg_logits = seg_logits[:, :, pad_left:pad_left + L]
        print(f"Cropped logits length: {seg_logits.shape[-1]}")

        # Pre-softmax P suppression
        if suppress_p:
            mask = torch.zeros(B, dtype=torch.bool, device=seg_logits.device)
            for i in range(B):
                if rid[i].item() in P_ABSENT_IDS:
                    mask[i] = True
            seg_logits[mask, 1, :] = -1e9  # Low value for P channel

        return seg_logits