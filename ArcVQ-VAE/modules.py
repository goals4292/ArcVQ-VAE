import torch
import torch.nn as nn
import torch.nn.functional as F

from quantise import VectorQuantiser

import numpy as np

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_channels):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=output_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    

class ArcLoss(nn.Module):
    def __init__(self, s=5.0, m=0.1, top_k=3):
        super().__init__()
        self.s = s
        self.m = m
        self.top_k = top_k

    def forward(self, z_e_x, codebook_weight):
        B, C, H, W = z_e_x.size()
        z_e_flat = z_e_x.permute(0, 2, 3, 1).reshape(-1, C)
        z_e_flat = F.normalize(z_e_flat, p=2, dim=1)

        e_norm = F.normalize(codebook_weight.detach(), p=2, dim=1)  # (K, C)

        cos_theta = torch.matmul(e_norm, z_e_flat.t())  # (K, BHW)

        # Top-K indices per codebook vector
        topk_val, topk_idx = torch.topk(cos_theta, self.top_k, dim=1)  # (K, top_k)

        # Apply margin to the top-K cosine values
        theta = torch.acos(torch.clamp(topk_val, -1.0 + 1e-7, 1.0 - 1e-7))
        cos_theta_m = torch.cos(theta + self.m)  # (K, top_k)

        # Clone logits
        logits = cos_theta.clone()

        # Replace the top-K values with margin-applied values
        row_idx = torch.arange(e_norm.size(0)).unsqueeze(1).expand(-1, self.top_k)  # (K, top_k)
        logits[row_idx, topk_idx] = cos_theta_m

        # Scale
        logits = self.s * logits  # (K, BHW)


        numerator = torch.sum(torch.exp(logits[row_idx, topk_idx]), dim=1)  # (K,)
        denominator = torch.sum(torch.exp(logits), dim=1)  # (K,)

        loss = -torch.log(numerator / denominator).mean()

        return loss



class Model(nn.Module):
    def __init__(self, input_dim, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost=0.25, use_arc_loss=True, arc_s=5.0, arc_m=0.1):
        super(Model, self).__init__()
        
        self._encoder = Encoder(input_dim, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        
        self._vq_vae = VectorQuantiser(num_embeddings, embedding_dim, commitment_cost)
        
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                input_dim)

        self.use_arc_loss = use_arc_loss
        if self.use_arc_loss:
            self.arc_loss = ArcLoss(s=arc_s, m=arc_m, top_k=3)


    def normalize_codebook(self):
        with torch.no_grad():
            weight = self._vq_vae.embedding.weight
            weight.data = F.normalize(weight.data, p=2, dim=1)
            #weight.data = F.normalize(weight.data, p=2, dim=1, eps=1e-12) * 5.0


    def clip_codebook_norms(self, step, alpha=0.0003):
        with torch.no_grad():
            weight = self._vq_vae.embedding.weight  # (K, D)
            norms = torch.norm(weight, p=2, dim=1, keepdim=True)  # (K, 1)

            max_norm = torch.exp(torch.tensor(alpha * step))  # scalar
            max_norm = max_norm.to(weight.device)  # (scalar tensor)
            
            clipped_weight = weight * (max_norm / norms)
            mask = (norms > max_norm).float()

            weight.data = mask * clipped_weight + (1 - mask) * weight


    def encode(self, x):
        z_e_x = self._encoder(x)
        z_e_x = self._pre_vq_conv(z_e_x)
        quantized, loss, _ = self._vq_vae(z_e_x)
        return loss, quantized

    def forward(self, x, step=None, gamma=None, gamma_decay=None):
        z_e_x = self._encoder(x)
        z_e_x = self._pre_vq_conv(z_e_x)

        if step is not None:
            self.clip_codebook_norms(step)

        #self.normalize_codebook()    
        quantized, vq_loss, (encodings, _) = self._vq_vae(z_e_x)

        if self.use_arc_loss:
            codebook_weight = self._vq_vae.embedding.weight
            arc_loss = self.arc_loss(z_e_x, codebook_weight)

            if gamma is not None and gamma_decay is not None and step is not None:
                weight = gamma * np.exp(-gamma_decay * step)
            else:
                weight = 0.0  # fallback default

            total_loss = vq_loss + weight * arc_loss
        else:
            arc_loss = None
            total_loss = vq_loss

        x_recon = self._decoder(quantized)
        return x_recon, total_loss, encodings, arc_loss
