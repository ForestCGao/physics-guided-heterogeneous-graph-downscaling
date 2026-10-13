import argparse
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from models.hgt import GNN
from models.spode import STRepresentationNetHRNoBatch
from utils.data import RadarDataset, collate_fn
from utils.losses import CompositeRadarLoss


def load_config(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def build_model(settings):
    gnn = GNN(
        in_dim=settings['in_dim'], n_hid=settings['hidden_dim'],
        num_types=2, num_relations=4, n_heads=settings['n_heads'],
        n_layers=settings['n_layers'], dropout=settings['dropout'],
        conv_name='dense_hgt', use_RTE=True,
    )
    return STRepresentationNetHRNoBatch(
        gnn_model=gnn, hidden_dim=settings['hidden_dim'],
        num_A_side=settings['num_A_side'], solver=settings['solver'],
        residual_alpha_init=settings['residual_alpha_init'],
    )


def predict(model, batch):
    return model(
        nf=batch['nf'], nt=batch['nt'], et=batch['et'], ei=batch['ei'],
        ety=batch['ety'], node_pos=batch['pos'], times=batch['times'],
        node_offsets=batch['offsets'],
    )


def make_loader(config, split, shuffle):
    data, training = config['data'], config['training']
    dataset = RadarDataset(data['root'], split, data, data['segment_len'], config['model']['num_A_side'])
    return DataLoader(
        dataset, batch_size=training['batch_size'], shuffle=shuffle,
        num_workers=training['num_workers'], pin_memory=True, collate_fn=collate_fn,
    )


def train_one_epoch(model, loader, criterion, optimizer, accelerator, settings, global_step):
    model.train()
    optimizer.zero_grad()
    total = 0.0
    count = 0
    for batch in loader:
        with accelerator.accumulate(model):
            pred = predict(model, batch)
            target = batch['target'].to(device=pred.device, dtype=pred.dtype)
            loss = criterion(pred, target, batch=batch)
            raw_loss = loss.detach().clone().item()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=settings['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
        total += raw_loss
        count += 1
        global_step += 1
    return total / max(count, 1), global_step


@torch.no_grad()
def validate(model, loader, criterion, accelerator, thresholds):
    model.eval()
    loss_sum, count = 0.0, 0
    totals = {thr: [0.0, 0.0, 0.0] for thr in thresholds}
    for batch in loader:
        pred = predict(model, batch)
        target = batch['target'].to(device=pred.device, dtype=pred.dtype)
        loss = criterion(pred, target, batch=batch)
        gathered = accelerator.gather_for_metrics(loss.detach().unsqueeze(0))
        loss_sum += gathered.mean().item()
        count += 1
        for thr in thresholds:
            p, y = pred > thr, target > thr
            stats = ((p & y).sum().float(), ((~p) & y).sum().float(), (p & (~y)).sum().float())
            for index, value in enumerate(stats):
                totals[thr][index] += accelerator.gather_for_metrics(value.unsqueeze(0)).sum().item()
    csi = {thr: (h / (h + m + f) if h + m + f else 1.0) for thr, (h, m, f) in totals.items()}
    return loss_sum / max(count, 1), csi


def main():
    parser = argparse.ArgumentParser(description='Train the paper HGT/ODE/GRU model.')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--data_root')
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config['data']['root'] = args.data_root
    settings = config['training']
    if settings['batch_size'] != 1:
        raise ValueError('batch_size must be 1; use grad_accum_steps for accumulation.')
    accelerator = Accelerator(
        mixed_precision=settings['mixed_precision'],
        gradient_accumulation_steps=settings['grad_accum_steps'],
    )
    set_seed(settings['seed'])
    train_loader = make_loader(config, config['data']['train_split'], True)
    val_loader = make_loader(config, config['data']['val_split'], False)
    model = build_model(config['model'])
    criterion = CompositeRadarLoss(**config['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], **settings['adam'])
    start_epoch, global_step = 1, 0
    best_val_loss, best_csi = float('inf'), float('-inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        best_csi = checkpoint.get('best_csi_0429', best_csi)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # As in the source, the patience counter starts at zero on resume.
    bad_count = 0
    thresholds = config['evaluation']['thresholds']
    for epoch in range(start_epoch, settings['epochs'] + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, accelerator,
            settings, global_step,
        )
        if epoch % settings['eval_interval']:
            continue
        val_loss, csi = validate(model, val_loader, criterion, accelerator, thresholds)
        improved = val_loss < best_val_loss - settings['early_stop_min_delta']
        if improved:
            best_val_loss, bad_count = val_loss, 0
        else:
            bad_count += 1
        best_csi = max(best_csi, csi[thresholds[-1]])
        if accelerator.is_main_process:
            checkpoint = dict(
                epoch=epoch, global_step=global_step,
                model=accelerator.unwrap_model(model).state_dict(),
                optimizer=optimizer.state_dict(), scheduler=None,
                best_val_loss=best_val_loss, best_csi_0429=best_csi, config=config,
            )
            torch.save(checkpoint, checkpoint_dir / f'model_{epoch:03d}.pt')
            torch.save(checkpoint, checkpoint_dir / 'latest.pt')
            if improved:
                torch.save(checkpoint, checkpoint_dir / 'best_val_loss_monitor.pt')
            print(f'Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}, CSI={csi}, patience={bad_count}')
        accelerator.wait_for_everyone()
        if 0 < settings['early_stop_patience'] <= bad_count:
            break


if __name__ == '__main__':
    main()
