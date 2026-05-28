import os, yaml, random, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from deeplens.projects.raw2task.sensors.optics import DifferentiablePSF as GaussianPSF
from deeplens.projects.raw2task.sensors.optics_deeplens import DeepLensPSF
from deeplens.projects.raw2task.sensors.cfa import LearnableCFA, superpixel_demux
from deeplens.projects.raw2task.sensors.noise import PoissonGaussianNoiseQuant
from deeplens.projects.raw2task.models.cnn_small import SmallCNN
from deeplens.projects.raw2task.utils.metrics import accuracy, AverageMeter


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class Sensor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.psf = DifferentiablePSF(init_sigma=cfg['sensor']['psf_init_sigma'], kernel_size=9)

        backend = cfg['sensor'].get('optics_backend', 'gaussian').lower()
        if backend == 'deeplens':
            self.psf = DeepLensPSF(cfg)
        elif backend == 'gaussian':
            self.psf = GaussianPSF(init_sigma=cfg['sensor']['psf_init_sigma'], kernel_size=9)
        else:
            raise ValueError(f"Unknown optics_backend: {backend}")


        self.cfa = LearnableCFA(init=cfg['sensor']['cfa_init'])
        self.exposure_logit = torch.nn.Parameter(torch.tensor(cfg['sensor']['exposure_init']))
        self.noise = PoissonGaussianNoiseQuant(
            bit_depth=cfg['sensor']['bit_depth'],
            read_noise_std=cfg['sensor']['read_noise_std'],
            shot_noise_scale=cfg['sensor']['shot_noise_scale'])
        self.make_superpixel = cfg['sensor']['make_superpixel']

    def forward(self, rgb):
        exposure = 0.25 + 3.75 * torch.sigmoid(self.exposure_logit)
        x = torch.clamp(rgb * exposure, 0, 1)
        x = self.psf(x)               # optics
        x = self.cfa(x)               # (B,1,H,W) mosaic RAW
        x = self.noise(x)             # noisy + quantized
        if self.make_superpixel:
            x = superpixel_demux(x)   # (B,4,H/2,W/2)
        return x

def get_loaders(cfg):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([ transforms.ToTensor() ])
    root = cfg['data']['root']
    tr = datasets.CIFAR10(root=root, train=True,  download=True, transform=tf_train)
    te = datasets.CIFAR10(root=root, train=False, download=True, transform=tf_test)
    return (DataLoader(tr, batch_size=cfg['data']['batch_size'], shuffle=True,
                       num_workers=cfg['data']['num_workers'], pin_memory=True),
            DataLoader(te, batch_size=512, shuffle=False,
                       num_workers=cfg['data']['num_workers'], pin_memory=True))

def main(config_path):
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed']); os.makedirs(cfg['train']['ckpt_dir'], exist_ok=True)
    train_loader, test_loader = get_loaders(cfg)

    sensor = Sensor(cfg).to(device)
    # freeze sensor params for fixed-camera baseline
    if bool(cfg['train'].get('freeze_sensor', False)):
        for p in sensor.parameters():
            p.requires_grad_(False)
    in_ch  = 4 if cfg['sensor']['make_superpixel'] else 1
    model  = SmallCNN(in_channels=in_ch, width=cfg['model']['width']).to(device)

    base_lr = float(cfg['train']['lr'])
    sensor_lr_mult = float(cfg['train'].get('sensor_lr_mult', 1.0))
    sensor_lr = base_lr * sensor_lr_mult

    weight_decay = float(cfg['train'].get('weight_decay', 1e-4))
    epochs = int(cfg['train'].get('epochs', 10))


    opt = optim.AdamW([
        {"params": model.parameters(), "lr": base_lr},
        {"params": sensor.parameters(), "lr": sensor_lr},
    ], weight_decay=weight_decay)

    crit = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, cfg['train']['epochs']+1):
        model.train(); sensor.train()
        lmeter, ameter = AverageMeter(), AverageMeter()
        for it, (img, tgt) in enumerate(train_loader, 1):
            img, tgt = img.to(device), tgt.to(device)
            raw = sensor(img); logits = model(raw); loss = crit(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            ameter.update(accuracy(logits.detach(), tgt), img.size(0))
            lmeter.update(loss.item(), img.size(0))
            if it % cfg['train']['log_interval'] == 0:
                print(f"Epoch {epoch} | {it}/{len(train_loader)} | loss {lmeter.avg:.4f} | acc {ameter.avg:.3f}")
        acc = evaluate(sensor, model, test_loader, device)
        print(f"[Eval] Epoch {epoch} | acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"sensor": sensor.state_dict(), "model": model.state_dict(), "cfg": cfg},
                       os.path.join(cfg['train']['ckpt_dir'], f"best_ep{epoch}_acc{best:.3f}.pt"))
            print("Saved best checkpoint.")
    print(f"Best Acc: {best:.4f}")

@torch.no_grad()
def evaluate(sensor, model, loader, device):
    sensor.eval(); model.eval()
    meter = AverageMeter()
    for img, tgt in loader:
        img, tgt = img.to(device), tgt.to(device)
        logits = model(sensor(img))
        meter.update(accuracy(logits, tgt), img.size(0))
    return meter.avg

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="deeplens/projects/raw2task/configs/cifar10_cls_baseline.yaml")
    a = p.parse_args()
    main(a.config)
