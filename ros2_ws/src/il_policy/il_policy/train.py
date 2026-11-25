"""Entrenamiento de Behavioral Cloning (MSE sobre (v, w) normalizados)."""

import argparse
import csv
from datetime import datetime
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from il_policy.contract import CONTRACT_VERSION, ContractConfig
from il_policy.dataset import ILDataset, load_npz, split_by_episode
from il_policy.evaluate import compute_metrics, measure_latency
from il_policy.model import MLPPolicy, count_parameters
from il_policy.policy import ILPolicy

DEFAULTS = {
    'contract': ContractConfig().to_dict(),
    'model': {'hidden': [128, 128, 64], 'dropout': 0.0},
    'training': {'epochs': 60, 'batch_size': 256, 'lr': 1e-3, 'weight_decay': 1e-4,
                 'val_frac': 0.15, 'test_frac': 0.15, 'patience': 10,
                 'seed': 42, 'device': 'cpu'},
}


def default_config_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('il_policy'), 'config', 'policy_mlp.yaml')
        if os.path.exists(path):
            return path
    except Exception:  # noqa: BLE001 — fuera de ROS no existe ament_index
        pass
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'policy_mlp.yaml')
    return path if os.path.exists(path) else None


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))  # copia profunda
    if path:
        import yaml
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for section, values in user.items():
            cfg.setdefault(section, {}).update(values or {})
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / max(n, 1)


def train(data_paths, out_dir, cfg_dict, verbose=True) -> dict:
    tcfg, mcfg = cfg_dict['training'], cfg_dict['model']
    contract = ContractConfig.from_dict(cfg_dict['contract'])
    set_seed(tcfg['seed'])
    device = torch.device(tcfg['device'])
    os.makedirs(out_dir, exist_ok=True)

    data = load_npz(data_paths, contract)
    tr, va, te = split_by_episode(data['episode_id'], tcfg['val_frac'], tcfg['test_frac'],
                                  tcfg['seed'])
    train_ds, val_ds = ILDataset(data, tr, contract), ILDataset(data, va, contract)
    train_dl = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4096)

    model = MLPPolicy(hidden=mcfg['hidden'], dropout=mcfg['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    if verbose:
        print(f'Muestras train/val/test: {len(tr)}/{len(va)}/{len(te)} '
              f'(episodios: {len(np.unique(data["episode_id"][tr]))}/'
              f'{len(np.unique(data["episode_id"][va]))}/'
              f'{len(np.unique(data["episode_id"][te]))}) | '
              f'parámetros: {count_parameters(model)}')

    ckpt_path = os.path.join(out_dir, 'best.pt')
    best, bad, history = float('inf'), 0, []
    for epoch in range(1, tcfg['epochs'] + 1):
        tl = run_epoch(model, train_dl, loss_fn, device, opt)
        vl = run_epoch(model, val_dl, loss_fn, device)
        sched.step(vl)
        history.append({'epoch': epoch, 'train_loss': tl, 'val_loss': vl,
                        'lr': opt.param_groups[0]['lr']})
        improved = vl < best - 1e-6
        if improved:
            best, bad = vl, 0
            torch.save({
                'contract_version': CONTRACT_VERSION,
                'contract': contract.to_dict(),
                'model_config': model.config,
                'model_state': model.state_dict(),
                'train_args': cfg_dict,
                'epoch': epoch,
                'created': datetime.now().isoformat(timespec='seconds'),
            }, ckpt_path)
        else:
            bad += 1
        if verbose:
            print(f'Época {epoch:3d} | train {tl:.5f} | val {vl:.5f}{"  *" if improved else ""}')
        if bad >= tcfg['patience']:
            if verbose:
                print(f'Early stopping en la época {epoch}')
            break

    # Métricas finales con el mejor modelo, en unidades físicas
    policy = ILPolicy.load(ckpt_path, device=str(device))
    metrics = {'best_val_loss': best,
               'val': compute_metrics(policy, data, va),
               'test': compute_metrics(policy, data, te) if len(te) else None}
    metrics.update(measure_latency(policy, n=200))

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt['metrics'] = metrics
    torch.save(ckpt, ckpt_path)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, 'history.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    if verbose and metrics['test']:
        t = metrics['test']
        print(f'\nTEST  MAE v = {t["mae_v"]:.4f} m/s (baseline {t["baseline_mae_v"]:.4f}) | '
              f'MAE w = {t["mae_w"]:.4f} rad/s (baseline {t["baseline_mae_w"]:.4f}) | '
              f'R² v = {t["r2_v"]:.3f}, R² w = {t["r2_w"]:.3f}')
        print(f'Latencia predict(): {metrics["latency_ms_mean"]:.3f} ms')
        print(f'Modelo guardado en {ckpt_path}')
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description='Entrena la política MLP de Behavioral Cloning')
    p.add_argument('--data', required=True, nargs='+', help='.npz o directorios con .npz')
    p.add_argument('--out', default='models/mlp_dummy')
    p.add_argument('--config', default=None, help='YAML (por defecto config/policy_mlp.yaml)')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--hidden', type=int, nargs='+', help='p. ej. --hidden 128 128 64')
    p.add_argument('--dropout', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', help='cpu o cuda')
    args = p.parse_args(argv)

    cfg = load_config(args.config or default_config_path())
    for key in ('epochs', 'batch_size', 'lr', 'patience', 'seed', 'device'):
        if getattr(args, key) is not None:
            cfg['training'][key] = getattr(args, key)
    if args.hidden:
        cfg['model']['hidden'] = args.hidden
    if args.dropout is not None:
        cfg['model']['dropout'] = args.dropout

    train(args.data, args.out, cfg)


if __name__ == '__main__':
    main()
