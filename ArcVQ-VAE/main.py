import os, random
import argparse
import multiprocessing as mp
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from modules import Model


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory,
                drop_last=False, worker_init_fn=None, generator=None,
                persistent_workers=True, prefetch_factor=4):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if num_workers > 0 and persistent_workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def train(data_loader, model, optimizer, args, writer, data_variance=1.0):
    model.train()

    for images, _ in data_loader:
        images = images.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        gamma_weight = args.gamma * np.exp(-args.gamma_decay * args.steps)

        x, total_loss, _, arc_loss = model(
            images, step=args.steps, gamma=args.gamma, gamma_decay=args.gamma_decay
        )

        loss_recons = F.mse_loss(x, images) / data_variance
        loss = loss_recons + total_loss  # total_loss = vq_loss + gamma * arcloss

        loss.backward()
        optimizer.step()

        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        if arc_loss is not None:
            writer.add_scalar('loss/train/arcloss', arc_loss.item(), args.steps)
            writer.add_scalar('weight/gamma', gamma_weight, args.steps)

        args.steps += 1


@torch.inference_mode()
def test(data_loader, model, args, writer, data_variance=1.0):
    model.eval()
    loss_recons, loss_total = 0.0, 0.0
    for images, _ in data_loader:
        images = images.to(args.device, non_blocking=True)
        x, total_loss, _, _ = model(images)
        loss_recons += F.mse_loss(x, images) / data_variance
        loss_total  += total_loss

    loss_recons /= len(data_loader)
    loss_total  /= len(data_loader)

    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization+arc', loss_total.item(), args.steps)

    return loss_recons.item(), loss_total.item()


@torch.inference_mode()
def generate_samples(images, model, args):
    model.eval()
    images = images.to(args.device, non_blocking=True)
    x, _, _, _ = model(images)
    return x


def main(args):
    writer = SummaryWriter(os.path.join(os.path.join(args.output_folder, 'logs'), args.exp_name))
    save_dir = os.path.join(os.path.join(args.output_folder, 'models'), args.exp_name)
    seed_everything(args.seed)

    data_variance = 1.0
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                                           download=True, transform=transform)
            test_dataset  = datasets.MNIST(args.data_folder, train=False,
                                           download=True, transform=transform)
            data_variance = np.var(train_dataset.data.numpy() / 255.0) * 4.0
            num_channels = 1

        elif args.dataset == 'fashion-mnist':
            train_dataset = datasets.FashionMNIST(args.data_folder, train=True,
                                                  download=True, transform=transform)
            test_dataset  = datasets.FashionMNIST(args.data_folder, train=False,
                                                  download=True, transform=transform)
            data_variance = np.var(train_dataset.data.numpy() / 255.0) * 4.0
            num_channels = 1

        elif args.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(args.data_folder, train=True,
                                             download=True, transform=transform)
            test_dataset  = datasets.CIFAR10(args.data_folder, train=False,
                                             download=True, transform=transform)
            data_variance = np.var(train_dataset.data / 255.0) * 4.0
            num_channels = 3

        valid_dataset = test_dataset
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    num_workers = min(args.num_workers, max(0, mp.cpu_count() // 2)) if args.tune_workers else args.num_workers
    use_persistent = args.persistent_workers and (num_workers > 0)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=use_persistent,
        prefetch_factor=args.prefetch_factor
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=use_persistent,
        prefetch_factor=args.prefetch_factor
    )

    viz_loader = make_loader(
        test_dataset,
        batch_size=min(32, args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=use_persistent,
        prefetch_factor=args.prefetch_factor
    )

    viz_images, _ = next(iter(viz_loader))


    model = Model(
        num_channels,
        args.hidden_size,
        args.num_residual_layers,
        args.num_residual_hidden,
        args.num_embedding,
        args.embedding_dim,
        args.commitment_cost,
        use_arc_loss=True,
        arc_s=5.0,
        arc_m=0.1
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer, data_variance)

        loss_rec, loss_vq = test(valid_loader, model, args, writer, data_variance)

        if (epoch + 1) % args.log_image_every == 0:
            rec_images = generate_samples(viz_images, model, args)
            input_grid = make_grid(viz_images, nrow=8, normalize=True)
            rec_grid   = make_grid(rec_images, nrow=8, normalize=True)
            writer.add_image('original', input_grid, epoch + 1)
            writer.add_image('reconstruction', rec_grid, epoch + 1)

        if (epoch == 0) or (loss_rec < best_loss):
            best_loss = loss_rec
            with open(os.path.join(save_dir, 'best.pt'), 'wb') as f:
                torch.save(model.state_dict(), f)

        with open(os.path.join(save_dir, f'model_{epoch+1}.pt'), 'wb') as f:
            torch.save(model.state_dict(), f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVQ-VAE')
    # General
    parser.add_argument('--data_folder', type=str, required=True, help='dataset root folder')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1024, help='train batch size')
    # Latent space
    parser.add_argument('--hidden_size', type=int, default=128, help='size of latent vectors')
    parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of residual layers')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers')
    # Quantiser parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of codebook entries')
    parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook entries')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='beta for commitment loss')
    # Optimization
    parser.add_argument('--seed', type=int, default=1, help="seed for reproducibility")
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for Adam')
    # I/O, logging
    parser.add_argument('--output_folder', type=str, default='./', help='output root folder')
    parser.add_argument('--exp_name', type=str, default='vqvae', help='experiment name')
    parser.add_argument('--log_image_every', type=int, default=5, help='log recon images every N epochs')
    parser.add_argument('--snapshot_every', type=int, default=0, help='save full snapshot every N epochs (0=off)')
    # DataLoader perf options
    parser.add_argument('--num_workers', type=int, default=max(0, mp.cpu_count() - 1),
                        help='number of DataLoader workers')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='keep DataLoader workers alive across epochs (recommended)')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='prefetch batches per worker (>=2)')
    parser.add_argument('--tune_workers', action='store_true',
                        help='auto-limit workers to ~half cores to reduce spawn overhead')
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')

    parser.add_argument('--gamma', type=float, default=1.0, help='weight for ArcLoss')
    parser.add_argument('--gamma_decay', type=float, default=5e-4, help='decay rate for ArcLoss')

    args = parser.parse_args()

    logs_dir = os.path.join(args.output_folder, 'logs')
    models_dir = os.path.join(args.output_folder, 'models')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.exp_name += f"-{os.environ['SLURM_JOB_ID']}"

    save_dir = os.path.join(models_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    args.steps = 0

    main(args)
