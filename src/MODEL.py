import torch
import torch.nn as nn
import torch.nn.functional as F
from CONFIG import EMBED_DIM, NUM_RHYTHMS, NUM_CLASSES, P_ABSENT_IDS

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, p_drop=0.0):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn: layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if p_drop > 0: layers.append(nn.Dropout(p_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ecg=1, embed_dim=EMBED_DIM, use_bn=False, p_drop=0.0):
        super().__init__()
        in_total = in_ecg + embed_dim
        filters = [32, 64, 128, 256, 512]  # Deeper for 5000 length
        self.rhythm_embed = nn.Embedding(NUM_RHYTHMS, embed_dim)

        # Encoder
        self.enc1 = ConvBlock1D(in_total, filters[0], use_bn, p_drop)
        self.enc2 = ConvBlock1D(filters[0], filters[1], use_bn, p_drop)
        self.enc3 = ConvBlock1D(filters[1], filters[2], use_bn, p_drop)
        self.enc4 = ConvBlock1D(filters[2], filters[3], use_bn, p_drop)
        self.enc5 = ConvBlock1D(filters[3], filters[4], use_bn, p_drop)
        self.pool = nn.MaxPool1d(2, 2)

        # Decoder
        self.up5 = nn.ConvTranspose1d(filters[4], filters[3], 2, 2)
        self.dec5 = ConvBlock1D(filters[3] + filters[3], filters[3], use_bn, p_drop)
        self.up4 = nn.ConvTranspose1d(filters[3], filters[2], 2, 2)
        self.dec4 = ConvBlock1D(filters[2] + filters[2], filters[2], use_bn, p_drop)
        self.up3 = nn.ConvTranspose1d(filters[2], filters[1], 2, 2)
        self.dec3 = ConvBlock1D(filters[1] + filters[1], filters[1], use_bn, p_drop)
        self.up2 = nn.ConvTranspose1d(filters[1], filters[0], 2, 2)
        self.dec2 = ConvBlock1D(filters[0] + filters[0], filters[0], use_bn, p_drop)
        self.out = nn.Conv1d(filters[0], NUM_CLASSES, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_ecg, rid, suppress_p=False):
        B, _, L = x_ecg.shape
        # Pad to multiple of 16 (for 4 pools)
        div_factor = 16
        pad_total = (div_factor - (L % div_factor)) % div_factor
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_ecg = F.pad(x_ecg, (pad_left, pad_right), mode='replicate')

        embed = self.rhythm_embed(rid).unsqueeze(-1).expand(-1, -1, x_ecg.size(2))
        x = torch.cat([x_ecg, embed], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Decoder
        d5 = self.up5(e5)
        e4 = F.interpolate(e4, size=d5.size(2), mode='linear', align_corners=False)
        d5 = self.dec5(torch.cat([d5, e4], dim=1))
        d4 = self.up4(d5)
        e3 = F.interpolate(e3, size=d4.size(2), mode='linear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)
        e2 = F.interpolate(e2, size=d3.size(2), mode='linear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        e1 = F.interpolate(e1, size=d2.size(2), mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        seg_logits = self.out(d2)

        # Pre-softmax P suppression if rhythm in P_ABSENT and suppress_p
        if suppress_p:
            mask = torch.zeros(B, dtype=torch.bool, device=seg_logits.device)
            for i in range(B):
                if rid[i].item() in P_ABSENT_IDS:
                    mask[i] = True
            seg_logits[mask, 1, :] = -1e9  # Low value for P channel

        # Crop back to original L
        seg_logits = seg_logits[:, :, pad_left:pad_left + L]

        return seg_logits