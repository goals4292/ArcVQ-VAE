import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantiser(nn.Module):
    def __init__(self, num_embed, embed_dim, beta):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embed, 1.0 / num_embed)

        with torch.no_grad():
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)

        self.used_code_indices = set()

    def forward(self, z):
        # z: (B, C, H, W) → (B*H*W, C)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z.view(-1, self.embed_dim)

        z_flat = F.normalize(z_flat, dim=1)

        e = F.normalize(self.embedding.weight, dim=1)
        dist = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(e ** 2, dim=1)
            - 2 * torch.matmul(z_flat, e.t())
        )  # (BHW, K)

        # get nearest embedding index
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)  # (BHW, 1)

        unique_codes = torch.unique(encoding_indices)
        num_unique = unique_codes.numel()
        with open("codebook_usage_log.txt", "a") as f:
            f.write(f"Unique codes used ({num_unique})\n")

        code_list = unique_codes.tolist()
        if isinstance(code_list, int):
            code_list = [code_list]
        self.used_code_indices.update(code_list)

        encodings = torch.zeros(encoding_indices.size(0), self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)


        # quantized vector
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)


        # losses
        loss_commit = F.mse_loss(z_q.detach(), z)
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss = loss_codebook + self.beta * loss_commit

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        # reshape back to (B, C, H, W)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, (encodings, encoding_indices.squeeze(1))
