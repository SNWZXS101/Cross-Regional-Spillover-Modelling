"""
Optimized V6 for the Guangdong-Guangxi two-region model.

Architecture:
    region-specific neural parameter functions -> differentiable mechanistic ODE -> reported cases

No neural state network is used. Therefore the fitted trajectory is already an ODE rollout.

Fixed:
    gamma = mu = 1/7
Trainable:
    beta
Correct observation model:
    Y_i(t) = q_i(t) * beta * c_i(t) * S_i(t) * I_i(t) / N_i

Expanded symmetric search bounds:
    c1,c2 : [1,16]
    q1,q2 : [0,1]
    m1,m2 : [0,0.2]
    d1,d2 : [0,0.2]
    beta  : [0.005,0.20]

Main changes relative to the first V6:
1. Region-specific 4-output parameter networks instead of one shared 8-output network.
2. Fourier time features for multi-wave temporal structure.
3. Multi-scale daily loss: raw + sqrt + log1p counts.
4. Much smaller cumulative-loss weight.
5. Moderate interior initialization of rate trajectories.
6. Screening is warm-started into the final fit.
7. Candidate score penalizes the worst-fitting region, not only the average.
8. Independent saved-rate RK4 validation remains the final acceptance criterion.
"""

# PyCharm revision copy: all caches and outputs remain under Revise_work.
import os as _os
from pathlib import Path as _Path
_group_dir = _Path(__file__).resolve().parent.parent
_cache = _group_dir / "logs" / "pycharm_runtime"
_cache.mkdir(parents=True, exist_ok=True)
for _sub in ("matplotlib", "temp", "numba"):
    (_cache / _sub).mkdir(exist_ok=True)
_os.environ["MPLCONFIGDIR"] = str(_cache / "matplotlib")
_os.environ["TEMP"] = _os.environ["TMP"] = str(_cache / "temp")
_os.environ["NUMBA_CACHE_DIR"] = str(_cache / "numba")
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


RATE_NAMES = ["c1", "c2", "q1", "q2", "m1", "m2", "d1", "d2"]
STATE_NAMES = ["S1", "S2", "Sq1", "Sq2", "I1", "I2", "Iq1", "Iq2", "R1", "R2", "C1", "C2"]


@dataclass
class Config:
    nu1_external: float = 0.0
    nu2_external: float = 0.0
    data_file: str = "Data_all_sm.xlsx"
    gd_column: str = "GD"
    gx_column: str = "GX"

    N1_raw: float = 12700e4
    N2_raw: float = 5037e4

    Sq10_raw: float = 0.0
    Sq20_raw: float = 4.0
    I10_raw: float = 0.0
    I20_raw: float = 1.0
    Iq10_raw: float = 0.0
    Iq20_raw: float = 0.0
    R10_raw: float = 0.0
    R20_raw: float = 0.0

    gamma_fixed: float = 1.0 / 7.0
    mu_fixed: float = 1.0 / 7.0

    beta_bounds: tuple = (0.005, 0.20)
    beta_init: float = 0.0755

    c1_bounds: tuple = (1.0, 16.0)
    c2_bounds: tuple = (1.0, 16.0)
    q1_bounds: tuple = (0.0, 1.0)
    q2_bounds: tuple = (0.0, 1.0)
    m1_bounds: tuple = (0.00, 0.2)
    m2_bounds: tuple = (0.00, 0.2)
    d1_bounds: tuple = (0.00, 0.2)
    d2_bounds: tuple = (0.00, 0.2)

    # Optimizer starting point only, not a fixed epidemiological assumption.
    c_init: float = 4.0
    q_init: float = 0.55
    m_init: float = 0.02
    d_init: float = 0.01

    rate_hidden_layers: int = 3
    rate_width: int = 48
    fourier_periods_days: tuple = (14.0, 28.0, 56.0, 112.0, 224.0)

    # CPU-friendly differentiable integration.
    screen_ode_substeps_per_day: int = 1
    ode_substeps_per_day: int = 1
    validation_rk4_substeps_per_day: int = 20

    count_scale_mode: str = "auto_cumulative"
    manual_count_scale: float = 1.0

    # Multi-scale daily loss. These sum to 1 by default.
    w_daily_raw: float = 0.30
    w_daily_sqrt: float = 0.40
    w_daily_log: float = 0.30
    w_cumulative: float = 0.05

    w_rate_smoothness: float = 3e-4
    w_rate_saturation: float = 1e-4
    saturation_logit_threshold: float = 3.5
    w_beta_saturation: float = 1e-5
    w_negative_state: float = 10.0

    seed: int = 2026
    adam_lr: float = 1.0e-3
    adam_iterations: int = 1000
    adam_print_every: int = 100
    scheduler_step: int = 700
    scheduler_gamma: float = 0.82
    grad_clip_norm: float = 100.0

    use_lbfgs: bool = True
    lbfgs_max_iter: int = 150
    lbfgs_lr: float = 0.6
    lbfgs_history_size: int = 75
    lbfgs_tolerance_grad: float = 1e-9
    lbfgs_tolerance_change: float = 1e-11

    run_autotune: bool = True
    screen_adam_iterations: int = 600
    screen_use_lbfgs: bool = False

    # seed, lr, smooth, saturation, beta0, cumulative, raw, sqrt, log
    autotune_candidates: tuple = (
        (2026,  1.0e-3, 3e-4, 1e-4, 0.0755, 0.05, 0.30, 0.40, 0.30),
        (6689,  8.0e-4, 1e-3, 3e-4, 0.0755, 0.05, 0.25, 0.45, 0.30),
        (31415, 1.2e-3, 3e-3, 3e-4, 0.1000, 0.10, 0.25, 0.40, 0.35),
    )

    worst_region_score_weight: float = 0.75
    boundary_score_weight: float = 0.25
    roughness_score_weight: float = 0.02
    boundary_fraction_threshold: float = 0.02

    device: str = "auto"
    dtype: str = "float32"
    cpu_num_threads: int = 0

    output_root: str = "Model4GDGXcq2x_v6_optimized01"
    case_name: str = "set1"


CONFIG = Config()


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(name):
    return torch.float64 if name.lower() == "float64" else torch.float32


def robust_scale_np(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    sd = float(np.std(x))
    mx = float(np.max(np.abs(x))) if x.size else 1.0
    return max(sd, 0.05 * mx, 1e-6)


def inv_sigmoid_fraction(x, low, high, eps=1e-6):
    z = (x - low) / (high - low)
    z = min(max(z, eps), 1.0 - eps)
    return math.log(z / (1.0 - z))


def fit_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y, p = y_true[mask], y_pred[mask]
    if len(y) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "NRMSE": np.inf}
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    den = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if den == 0 else 1.0 - float(np.sum((y - p) ** 2)) / den
    nrmse = rmse / max(float(np.std(y)), 1e-8)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NRMSE": nrmse}

class FourierTimeEncoder(nn.Module):
    def __init__(self, periods_days, t_max, dtype):
        super().__init__()
        self.register_buffer(
            "periods", torch.tensor(periods_days, dtype=dtype).view(1, -1)
        )
        self.t_max = float(t_max)

    @property
    def output_dim(self):
        return 1 + 2 * self.periods.shape[1]

    def forward(self, t):
        t_norm = 2.0 * t / self.t_max - 1.0
        angle = 2.0 * math.pi * t / self.periods
        return torch.cat([t_norm, torch.sin(angle), torch.cos(angle)], dim=1)


class RegionParameterMLP(nn.Module):
    """Outputs raw logits for c,q,m,d for one region."""
    def __init__(self, input_dim, hidden_layers, width, bounds4, initial4):
        super().__init__()
        layers = []
        last = input_dim
        for _ in range(hidden_layers):
            lin = nn.Linear(last, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.Tanh()])
            last = width
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(last, 4)

        # Start from constant, moderate, interior trajectories.
        nn.init.zeros_(self.out.weight)
        bias = [
            inv_sigmoid_fraction(v, b[0], b[1])
            for v, b in zip(initial4, bounds4)
        ]
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor(bias, dtype=self.out.bias.dtype))

    def forward(self, z):
        return self.out(self.hidden(z))


class ParameterNet(nn.Module):
    def __init__(self, cfg, t_max, dtype):
        super().__init__()
        self.cfg = cfg
        self.encoder = FourierTimeEncoder(cfg.fourier_periods_days, t_max, dtype)
        init4 = [cfg.c_init, cfg.q_init, cfg.m_init, cfg.d_init]

        self.region1 = RegionParameterMLP(
            self.encoder.output_dim, cfg.rate_hidden_layers, cfg.rate_width,
            [cfg.c1_bounds, cfg.q1_bounds, cfg.m1_bounds, cfg.d1_bounds], init4
        )
        self.region2 = RegionParameterMLP(
            self.encoder.output_dim, cfg.rate_hidden_layers, cfg.rate_width,
            [cfg.c2_bounds, cfg.q2_bounds, cfg.m2_bounds, cfg.d2_bounds], init4
        )

        lows = [
            cfg.c1_bounds[0], cfg.c2_bounds[0], cfg.q1_bounds[0], cfg.q2_bounds[0],
            cfg.m1_bounds[0], cfg.m2_bounds[0], cfg.d1_bounds[0], cfg.d2_bounds[0]
        ]
        highs = [
            cfg.c1_bounds[1], cfg.c2_bounds[1], cfg.q1_bounds[1], cfg.q2_bounds[1],
            cfg.m1_bounds[1], cfg.m2_bounds[1], cfg.d1_bounds[1], cfg.d2_bounds[1]
        ]
        self.register_buffer("rate_lows", torch.tensor(lows, dtype=dtype).view(1, 8))
        self.register_buffer("rate_highs", torch.tensor(highs, dtype=dtype).view(1, 8))

        beta_raw0 = inv_sigmoid_fraction(
            cfg.beta_init, cfg.beta_bounds[0], cfg.beta_bounds[1]
        )
        self.beta_raw = nn.Parameter(torch.tensor([beta_raw0], dtype=dtype))

    def raw_rates(self, t):
        z = self.encoder(t)
        a = self.region1(z)  # c1,q1,m1,d1
        b = self.region2(z)  # c2,q2,m2,d2
        return torch.cat(
            [a[:, 0:1], b[:, 0:1], a[:, 1:2], b[:, 1:2],
             a[:, 2:3], b[:, 2:3], a[:, 3:4], b[:, 3:4]], dim=1
        )

    def rates_tensor(self, t):
        raw = self.raw_rates(t)
        return self.rate_lows + (self.rate_highs - self.rate_lows) * torch.sigmoid(raw)

    def beta(self):
        lo, hi = self.cfg.beta_bounds
        return lo + (hi - lo) * torch.sigmoid(self.beta_raw)


class DifferentiableTwoRegionODE(nn.Module):
    def __init__(self, cfg, pnet, N1_scaled, N2_scaled, x0_scaled, dtype, device):
        super().__init__()
        self.cfg = cfg
        self.pnet = pnet
        self.N1 = float(N1_scaled)
        self.N2 = float(N2_scaled)
        self.dtype_ = dtype
        self.device_ = device
        self.register_buffer("gamma", torch.tensor(cfg.gamma_fixed, dtype=dtype))
        self.register_buffer("mu", torch.tensor(cfg.mu_fixed, dtype=dtype))
        self.register_buffer("x0", torch.tensor(x0_scaled, dtype=dtype).view(1, 12))

    def rhs_with_rates(self, x, rates, beta):
        S1, S2, Sq1, Sq2, I1, I2, Iq1, Iq2, R1, R2, C1, C2 = [
            x[:, i:i+1] for i in range(12)
        ]
        c1, c2, q1, q2, m1, m2, d1, d2 = [rates[:, i:i+1] for i in range(8)]

        lam1 = c1 * S1 * I1 / self.N1
        lam2 = c2 * S2 * I2 / self.N2

        dS1 = -lam1 + (1-q1)*(1-beta)*(1-m1)*lam1 + self.mu*Sq1 + (1-q2)*(1-beta)*m2*lam2
        dS2 = -lam2 + (1-q2)*(1-beta)*(1-m2)*lam2 + self.mu*Sq2 + (1-q1)*(1-beta)*m1*lam1
        dSq1 = q1*(1-beta)*lam1 - self.mu*Sq1
        dSq2 = q2*(1-beta)*lam2 - self.mu*Sq2
        dI1 = (1-q1)*beta*(1-m1)*lam1 - self.gamma*I1 + (1-q2)*beta*m2*lam2 - d1*I1 + d2*I2
        dI2 = (1-q2)*beta*(1-m2)*lam2 - self.gamma*I2 + (1-q1)*beta*m1*lam1 - d2*I2 + d1*I1
        dIq1 = q1*beta*lam1 - self.gamma*Iq1
        dIq2 = q2*beta*lam2 - self.gamma*Iq2
        dR1 = self.gamma*(I1 + Iq1)
        dR2 = self.gamma*(I2 + Iq2)
        dC1 = q1*beta*lam1
        dC2 = q2*beta*lam2

        # External infection among local residents, with no added population.
        u1 = self.cfg.nu1_external * S1
        u2 = self.cfg.nu2_external * S2
        dS1 = dS1 - u1; dS2 = dS2 - u2
        dI1 = dI1 + (1-q1)*u1; dI2 = dI2 + (1-q2)*u2
        dIq1 = dIq1 + q1*u1; dIq2 = dIq2 + q2*u2
        dC1 = dC1 + q1*u1; dC2 = dC2 + q2*u2

        return torch.cat(
            [dS1, dS2, dSq1, dSq2, dI1, dI2, dIq1, dIq2, dR1, dR2, dC1, dC2], dim=1
        )

    def precompute_fine_rates(self, n_days, substeps):
        half = 2 * substeps
        n_fine = n_days * half + 1
        t = torch.arange(n_fine, dtype=self.dtype_, device=self.device_).view(-1, 1) / float(half)
        return self.pnet.rates_tensor(t), self.pnet.raw_rates(t)

    def rollout(self, n_days, substeps):
        half = 2 * substeps
        h = 1.0 / substeps
        rates_fine, raw_fine = self.precompute_fine_rates(n_days, substeps)
        beta = self.pnet.beta()
        x = self.x0
        states = [x]

        for day in range(n_days):
            for sub in range(substeps):
                base = day * half + 2 * sub
                r1 = rates_fine[base:base+1]
                rm = rates_fine[base+1:base+2]
                r4 = rates_fine[base+2:base+3]

                k1 = self.rhs_with_rates(x, r1, beta)
                k2 = self.rhs_with_rates(x + 0.5*h*k1, rm, beta)
                k3 = self.rhs_with_rates(x + 0.5*h*k2, rm, beta)
                k4 = self.rhs_with_rates(x + h*k3, r4, beta)
                x = x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            states.append(x)

        states_day = torch.cat(states, dim=0)
        C1, C2 = states_day[:, 10], states_day[:, 11]
        daily = torch.stack([C1[1:] - C1[:-1], C2[1:] - C2[:-1]], dim=1)

        t_daily = torch.arange(n_days, dtype=self.dtype_, device=self.device_).view(-1, 1)
        rates_daily = self.pnet.rates_tensor(t_daily)
        raw_daily = self.pnet.raw_rates(t_daily)
        return states_day, daily, rates_daily, raw_daily

class ODEFitLoss:
    def __init__(self, cfg, solver, data):
        self.cfg = cfg
        self.solver = solver
        self.data = data

    @staticmethod
    def mse(x):
        return torch.mean(x*x)

    def daily_multiscale(self, pred_scaled):
        scale = self.data["count_scale"]
        p1, p2 = pred_scaled[:, 0:1], pred_scaled[:, 1:2]

        raw = (
            self.mse((p1 - self.data["y1_t"]) / self.data["sy1"])
            + self.mse((p2 - self.data["y2_t"]) / self.data["sy2"])
        )

        p1_raw = torch.clamp(p1 * scale, min=0.0)
        p2_raw = torch.clamp(p2 * scale, min=0.0)
        sqrt1, sqrt2 = torch.sqrt(p1_raw + 1.0), torch.sqrt(p2_raw + 1.0)
        log1, log2 = torch.log1p(p1_raw), torch.log1p(p2_raw)

        sqrt_loss = (
            self.mse((sqrt1 - self.data["sqrt_y1_t"]) / self.data["sqrt_sy1"])
            + self.mse((sqrt2 - self.data["sqrt_y2_t"]) / self.data["sqrt_sy2"])
        )
        log_loss = (
            self.mse((log1 - self.data["log_y1_t"]) / self.data["log_sy1"])
            + self.mse((log2 - self.data["log_y2_t"]) / self.data["log_sy2"])
        )

        combined = (
            self.cfg.w_daily_raw * raw
            + self.cfg.w_daily_sqrt * sqrt_loss
            + self.cfg.w_daily_log * log_loss
        )
        return combined, raw, sqrt_loss, log_loss

    def compute(self, n_days, substeps):
        states, daily, rates, raw_logits = self.solver.rollout(n_days, substeps)
        daily_combined, raw_loss, sqrt_loss, log_loss = self.daily_multiscale(daily)

        predC1, predC2 = states[1:, 10:11], states[1:, 11:12]
        cumulative = (
            self.mse((predC1 - self.data["C1_t"]) / self.data["sC1"])
            + self.mse((predC2 - self.data["C2_t"]) / self.data["sC2"])
        )

        lows, highs = self.solver.pnet.rate_lows, self.solver.pnet.rate_highs
        frac = (rates - lows) / (highs - lows)
        if len(frac) >= 3:
            second = frac[2:] - 2*frac[1:-1] + frac[:-2]
            smoothness = self.mse(second)
        else:
            smoothness = torch.tensor(0.0, dtype=states.dtype, device=states.device)

        excess = torch.relu(torch.abs(raw_logits) - self.cfg.saturation_logit_threshold)
        saturation = self.mse(excess)
        beta_excess = torch.relu(
            torch.abs(self.solver.pnet.beta_raw) - self.cfg.saturation_logit_threshold
        )
        beta_saturation = self.mse(beta_excess)
        negative_state = self.mse(torch.relu(-states[:, :10]))

        total = (
            daily_combined
            + self.cfg.w_cumulative * cumulative
            + self.cfg.w_rate_smoothness * smoothness
            + self.cfg.w_rate_saturation * saturation
            + self.cfg.w_beta_saturation * beta_saturation
            + self.cfg.w_negative_state * negative_state
        )
        parts = {
            "daily_combined": daily_combined,
            "daily_raw": raw_loss,
            "daily_sqrt": sqrt_loss,
            "daily_log": log_loss,
            "cumulative": cumulative,
            "smoothness": smoothness,
            "saturation": saturation,
            "negative_state": negative_state,
        }
        return total, parts, states, daily, rates


def train_candidate(cfg, data, out_dir, is_screen=False, warm_start_state_dict=None):
    set_all_seeds(cfg.seed)
    pnet = ParameterNet(cfg, float(data["n_days"]), data["dtype"]).to(
        device=data["device"], dtype=data["dtype"]
    )
    if warm_start_state_dict is not None:
        pnet.load_state_dict(warm_start_state_dict)

    solver = DifferentiableTwoRegionODE(
        cfg, pnet, data["N1_scaled"], data["N2_scaled"], data["x0_scaled"],
        data["dtype"], data["device"]
    ).to(device=data["device"], dtype=data["dtype"])
    loss_obj = ODEFitLoss(cfg, solver, data)

    iterations = cfg.screen_adam_iterations if is_screen else cfg.adam_iterations
    substeps = cfg.screen_ode_substeps_per_day if is_screen else cfg.ode_substeps_per_day
    use_lbfgs = cfg.screen_use_lbfgs if is_screen else cfg.use_lbfgs

    opt = torch.optim.Adam(pnet.parameters(), lr=cfg.adam_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma
    )
    rows, start = [], time.time()
    tag = "SCREEN" if is_screen else "FINAL"
    print(
        f"\n===== {tag} | seed={cfg.seed} | lr={cfg.adam_lr:g} | "
        f"substeps/day={substeps} | cum={cfg.w_cumulative:g} | "
        f"daily(raw/sqrt/log)=({cfg.w_daily_raw:g}/{cfg.w_daily_sqrt:g}/{cfg.w_daily_log:g}) ====="
    )

    for it in range(iterations + 1):
        opt.zero_grad(set_to_none=True)
        total, parts, _, _, _ = loss_obj.compute(data["n_days"], substeps)
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite training loss.")
        total.backward()
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(pnet.parameters(), cfg.grad_clip_norm)
        opt.step()
        scheduler.step()

        if it % cfg.adam_print_every == 0 or it == iterations:
            row = {
                "iteration": it,
                "total": float(total.detach().cpu()),
                "daily_combined": float(parts["daily_combined"].detach().cpu()),
                "daily_raw": float(parts["daily_raw"].detach().cpu()),
                "daily_sqrt": float(parts["daily_sqrt"].detach().cpu()),
                "daily_log": float(parts["daily_log"].detach().cpu()),
                "cumulative": float(parts["cumulative"].detach().cpu()),
                "smoothness": float(parts["smoothness"].detach().cpu()),
                "saturation": float(parts["saturation"].detach().cpu()),
                "negative_state": float(parts["negative_state"].detach().cpu()),
                "beta": float(pnet.beta().detach().cpu()),
                "lr": opt.param_groups[0]["lr"],
                "elapsed_sec": time.time() - start,
            }
            rows.append(row)
            print(
                f"It {it:5d} | total={row['total']:.4e} | daily={row['daily_combined']:.3e} "
                f"[raw={row['daily_raw']:.2e},sqrt={row['daily_sqrt']:.2e},log={row['daily_log']:.2e}] | "
                f"cum={row['cumulative']:.2e} | sat={row['saturation']:.2e} | beta={row['beta']:.6f}"
            )

    if use_lbfgs:
        print("\n===== L-BFGS refinement =====")
        lbfgs = torch.optim.LBFGS(
            pnet.parameters(), lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_max_iter,
            max_eval=int(cfg.lbfgs_max_iter * 1.25), history_size=cfg.lbfgs_history_size,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe",
        )
        counter = {"n": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            total, parts, _, _, _ = loss_obj.compute(data["n_days"], substeps)
            total.backward()
            counter["n"] += 1
            if counter["n"] == 1 or counter["n"] % 25 == 0:
                print(
                    f"LBFGS {counter['n']:4d} | total={float(total.detach().cpu()):.4e} | "
                    f"daily={float(parts['daily_combined'].detach().cpu()):.3e} | "
                    f"cum={float(parts['cumulative'].detach().cpu()):.3e} | "
                    f"sat={float(parts['saturation'].detach().cpu()):.2e} | "
                    f"beta={float(pnet.beta().detach().cpu()):.6f}"
                )
            return total
        lbfgs.step(closure)

    with torch.no_grad():
        total, parts, states, daily, rates = loss_obj.compute(data["n_days"], substeps)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "loss_history.csv", index=False)

    return {
        "pnet": pnet,
        "pnet_state_dict": deepcopy(pnet.state_dict()),
        "states_scaled": states.detach().cpu().numpy(),
        "daily_scaled": daily.detach().cpu().numpy(),
        "rates_daily": rates.detach().cpu().numpy(),
        "beta": float(pnet.beta().detach().cpu()),
        "training_seconds": time.time() - start,
        "final_loss": float(total.detach().cpu()),
        "loss_parts": {k: float(v.detach().cpu()) for k, v in parts.items()},
        "substeps": substeps,
    }

# =============================================================================
# Independent saved-rate validation
# =============================================================================

def interp_daily(arr, t):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if t <= 0:
        return float(arr[0])
    if t >= len(arr) - 1:
        return float(arr[-1])
    j = int(math.floor(t))
    a = t - j
    return float((1-a)*arr[j] + a*arr[j+1])


def numpy_rhs(t, x, rates, beta, gamma, mu, N1, N2, nu1, nu2):
    S1,S2,Sq1,Sq2,I1,I2,Iq1,Iq2,R1,R2,C1,C2 = x
    c1 = interp_daily(rates["c1"], t)
    c2 = interp_daily(rates["c2"], t)
    q1 = interp_daily(rates["q1"], t)
    q2 = interp_daily(rates["q2"], t)
    m1 = interp_daily(rates["m1"], t)
    m2 = interp_daily(rates["m2"], t)
    d1 = interp_daily(rates["d1"], t)
    d2 = interp_daily(rates["d2"], t)

    lam1 = c1*S1*I1/N1
    lam2 = c2*S2*I2/N2
    dS1 = -lam1 + (1-q1)*(1-beta)*(1-m1)*lam1 + mu*Sq1 + (1-q2)*(1-beta)*m2*lam2
    dS2 = -lam2 + (1-q2)*(1-beta)*(1-m2)*lam2 + mu*Sq2 + (1-q1)*(1-beta)*m1*lam1
    dSq1 = q1*(1-beta)*lam1 - mu*Sq1
    dSq2 = q2*(1-beta)*lam2 - mu*Sq2
    dI1 = (1-q1)*beta*(1-m1)*lam1 - gamma*I1 + (1-q2)*beta*m2*lam2 - d1*I1 + d2*I2
    dI2 = (1-q2)*beta*(1-m2)*lam2 - gamma*I2 + (1-q1)*beta*m1*lam1 - d2*I2 + d1*I1
    dIq1 = q1*beta*lam1 - gamma*Iq1
    dIq2 = q2*beta*lam2 - gamma*Iq2
    dR1 = gamma*(I1 + Iq1)
    dR2 = gamma*(I2 + Iq2)
    dC1 = q1*beta*lam1
    dC2 = q2*beta*lam2
    u1, u2 = nu1*S1, nu2*S2
    dS1 -= u1; dS2 -= u2
    dI1 += (1-q1)*u1; dI2 += (1-q2)*u2
    dIq1 += q1*u1; dIq2 += q2*u2
    dC1 += q1*u1; dC2 += q2*u2
    return np.array(
        [dS1,dS2,dSq1,dSq2,dI1,dI2,dIq1,dIq2,dR1,dR2,dC1,dC2], dtype=float
    )


def independent_rk4_rollout(cfg, data, result):
    rates = {name: result["rates_daily"][:, i] for i, name in enumerate(RATE_NAMES)}
    x = np.asarray(data["x0_raw"], dtype=float).copy()
    states = [x.copy()]
    substeps = cfg.validation_rk4_substeps_per_day
    h = 1.0 / substeps
    warnings = []

    for day in range(data["n_days"]):
        for sub in range(substeps):
            t = day + sub*h
            k1 = numpy_rhs(t, x, rates, result["beta"], cfg.gamma_fixed, cfg.mu_fixed, cfg.N1_raw, cfg.N2_raw, cfg.nu1_external, cfg.nu2_external)
            k2 = numpy_rhs(t+0.5*h, x+0.5*h*k1, rates, result["beta"], cfg.gamma_fixed, cfg.mu_fixed, cfg.N1_raw, cfg.N2_raw, cfg.nu1_external, cfg.nu2_external)
            k3 = numpy_rhs(t+0.5*h, x+0.5*h*k2, rates, result["beta"], cfg.gamma_fixed, cfg.mu_fixed, cfg.N1_raw, cfg.N2_raw, cfg.nu1_external, cfg.nu2_external)
            k4 = numpy_rhs(t+h, x+h*k3, rates, result["beta"], cfg.gamma_fixed, cfg.mu_fixed, cfg.N1_raw, cfg.N2_raw, cfg.nu1_external, cfg.nu2_external)
            x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            if not np.all(np.isfinite(x)):
                warnings.append(f"Non-finite state at t={t:.4f}")
                break
            if np.min(x[:10]) < -1e-5:
                warnings.append(f"Negative epidemiological state at t={t:.4f}: min={np.min(x[:10]):.6g}")
        states.append(x.copy())

    states = np.asarray(states)
    p1 = states[1:, 10] - states[:-1, 10]
    p2 = states[1:, 11] - states[:-1, 11]
    return states, p1, p2, warnings


def boundary_and_roughness(rates, cfg):
    lows = np.array([
        cfg.c1_bounds[0], cfg.c2_bounds[0], cfg.q1_bounds[0], cfg.q2_bounds[0],
        cfg.m1_bounds[0], cfg.m2_bounds[0], cfg.d1_bounds[0], cfg.d2_bounds[0]
    ])
    highs = np.array([
        cfg.c1_bounds[1], cfg.c2_bounds[1], cfg.q1_bounds[1], cfg.q2_bounds[1],
        cfg.m1_bounds[1], cfg.m2_bounds[1], cfg.d1_bounds[1], cfg.d2_bounds[1]
    ])
    frac = (rates - lows) / (highs - lows)
    thr = cfg.boundary_fraction_threshold
    boundary = float(np.mean((frac < thr) | (frac > 1-thr)))
    per_rate = {
        RATE_NAMES[j]: float(np.mean((frac[:, j] < thr) | (frac[:, j] > 1-thr)))
        for j in range(8)
    }
    roughness = 0.0
    if len(frac) >= 3:
        second = frac[2:] - 2*frac[1:-1] + frac[:-2]
        roughness = float(np.mean(np.abs(second)))
    return boundary, roughness, per_rate


def evaluate_candidate(cfg, data, result):
    states, p1, p2, warnings = independent_rk4_rollout(cfg, data, result)
    gd = fit_metrics(data["y1_raw"], p1)
    gx = fit_metrics(data["y2_raw"], p2)
    boundary, roughness, per_rate = boundary_and_roughness(result["rates_daily"], cfg)
    worst = max(gd["NRMSE"], gx["NRMSE"])
    score = (
        gd["NRMSE"] + gx["NRMSE"]
        + cfg.worst_region_score_weight * worst
        + cfg.boundary_score_weight * boundary
        + cfg.roughness_score_weight * roughness
    )
    return {
        "score": float(score), "GD": gd, "GX": gx,
        "boundary_fraction": boundary, "roughness": roughness,
        "per_rate_boundary": per_rate,
        "independent_states_raw": states,
        "independent_daily1_raw": p1,
        "independent_daily2_raw": p2,
        "warnings": warnings,
    }


# =============================================================================
# Data, saving, plots
# =============================================================================

def prepare_data(cfg):
    device = choose_device(cfg.device)
    dtype = choose_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)
    if cfg.cpu_num_threads > 0 and device.type == "cpu":
        torch.set_num_threads(cfg.cpu_num_threads)

    df = pd.read_excel(cfg.data_file)
    y1_raw = df[cfg.gd_column].to_numpy(dtype=np.float64).reshape(-1)
    y2_raw = df[cfg.gx_column].to_numpy(dtype=np.float64).reshape(-1)
    if len(y1_raw) != len(y2_raw):
        raise ValueError("GD/GX data lengths differ.")

    n = len(y1_raw)
    C1_raw, C2_raw = np.cumsum(y1_raw), np.cumsum(y2_raw)
    scale = (
        max(float(C1_raw[-1]), float(C2_raw[-1]), 1.0)
        if cfg.count_scale_mode == "auto_cumulative"
        else float(cfg.manual_count_scale)
    )

    N1, N2 = cfg.N1_raw/scale, cfg.N2_raw/scale
    Sq10, Sq20 = cfg.Sq10_raw/scale, cfg.Sq20_raw/scale
    I10, I20 = cfg.I10_raw/scale, cfg.I20_raw/scale
    Iq10, Iq20 = cfg.Iq10_raw/scale, cfg.Iq20_raw/scale
    R10, R20 = cfg.R10_raw/scale, cfg.R20_raw/scale
    S10 = N1 - Sq10 - I10 - Iq10 - R10
    S20 = N2 - Sq20 - I20 - Iq20 - R20
    x0_scaled = np.array([S10,S20,Sq10,Sq20,I10,I20,Iq10,Iq20,R10,R20,0.0,0.0])
    x0_raw = x0_scaled * scale

    y1_scaled, y2_scaled = y1_raw/scale, y2_raw/scale
    C1_scaled, C2_scaled = C1_raw/scale, C2_raw/scale
    sqrt_y1, sqrt_y2 = np.sqrt(y1_raw+1.0), np.sqrt(y2_raw+1.0)
    log_y1, log_y2 = np.log1p(y1_raw), np.log1p(y2_raw)

    def Tcol(x):
        return torch.tensor(np.asarray(x).reshape(-1,1), dtype=dtype, device=device)

    return {
        "device": device, "dtype": dtype, "n_days": n, "count_scale": scale,
        "N1_scaled": N1, "N2_scaled": N2, "x0_scaled": x0_scaled, "x0_raw": x0_raw,
        "y1_raw": y1_raw, "y2_raw": y2_raw, "C1_raw": C1_raw, "C2_raw": C2_raw,
        "y1_t": Tcol(y1_scaled), "y2_t": Tcol(y2_scaled),
        "C1_t": Tcol(C1_scaled), "C2_t": Tcol(C2_scaled),
        "sy1": torch.tensor(robust_scale_np(y1_scaled), dtype=dtype, device=device),
        "sy2": torch.tensor(robust_scale_np(y2_scaled), dtype=dtype, device=device),
        "sC1": torch.tensor(robust_scale_np(C1_scaled), dtype=dtype, device=device),
        "sC2": torch.tensor(robust_scale_np(C2_scaled), dtype=dtype, device=device),
        "sqrt_y1_t": Tcol(sqrt_y1), "sqrt_y2_t": Tcol(sqrt_y2),
        "sqrt_sy1": torch.tensor(robust_scale_np(sqrt_y1), dtype=dtype, device=device),
        "sqrt_sy2": torch.tensor(robust_scale_np(sqrt_y2), dtype=dtype, device=device),
        "log_y1_t": Tcol(log_y1), "log_y2_t": Tcol(log_y2),
        "log_sy1": torch.tensor(robust_scale_np(log_y1), dtype=dtype, device=device),
        "log_sy2": torch.tensor(robust_scale_np(log_y2), dtype=dtype, device=device),
    }


def save_vector(path, x):
    np.savetxt(path, np.asarray(x).reshape(-1,1))


def save_outputs(cfg, data, result, val, out_dir):
    out = Path(out_dir)
    scale = data["count_scale"]
    states_raw = result["states_scaled"] * scale
    daily_raw = result["daily_scaled"] * scale
    for j, key in enumerate(STATE_NAMES[:10]):
        save_vector(out/f"{key}.txt", states_raw[:-1, j])
    save_vector(out/"I1_new.txt", daily_raw[:,0])
    save_vector(out/"I2_new.txt", daily_raw[:,1])
    save_vector(out/"I1_sum.txt", states_raw[1:,10])
    save_vector(out/"I2_sum.txt", states_raw[1:,11])
    for j, key in enumerate(RATE_NAMES):
        save_vector(out/f"{key}.txt", result["rates_daily"][:,j])
    save_vector(out/"beta.txt", [result["beta"]])
    save_vector(out/"gamma.txt", [cfg.gamma_fixed])
    save_vector(out/"mu.txt", [cfg.mu_fixed])
    save_vector(out/"COUNT_SCALE.txt", [scale])
    save_vector(out/"Independent_I1_new.txt", val["independent_daily1_raw"])
    save_vector(out/"Independent_I2_new.txt", val["independent_daily2_raw"])
    save_vector(out/"Independent_I1_sum.txt", val["independent_states_raw"][1:,10])
    save_vector(out/"Independent_I2_sum.txt", val["independent_states_raw"][1:,11])


def save_plots(data, result, val, out_dir):
    out = Path(out_dir)
    t = np.arange(data["n_days"])
    fit = result["daily_scaled"] * data["count_scale"]
    fig, ax = plt.subplots(2,1,figsize=(13,8),sharex=True)
    ax[0].plot(t,data["y1_raw"],"k.",ms=4,label="GD observed")
    ax[0].plot(t,fit[:,0],lw=2,label="Differentiable ODE fit")
    ax[0].plot(t,val["independent_daily1_raw"],"--",lw=1.5,label="Independent saved-rate rollout")
    ax[0].set_ylabel("Daily reported cases"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(t,data["y2_raw"],"g.",ms=4,label="GX observed")
    ax[1].plot(t,fit[:,1],lw=2,label="Differentiable ODE fit")
    ax[1].plot(t,val["independent_daily2_raw"],"--",lw=1.5,label="Independent saved-rate rollout")
    ax[1].set_ylabel("Daily reported cases"); ax[1].set_xlabel("Days"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(out/"I_fitting.png",dpi=300); plt.close(fig)

    fig, axes = plt.subplots(4,1,figsize=(13,14),sharex=True)
    groups = [(0,1,"Contact rate"),(2,3,"Quarantine/detection fraction"),(4,5,"Cross-region mixing fraction"),(6,7,"Direct infected movement rate")]
    for axx,(a,b,label) in zip(axes,groups):
        axx.plot(t,result["rates_daily"][:,a],lw=2,label=RATE_NAMES[a])
        axx.plot(t,result["rates_daily"][:,b],lw=2,label=RATE_NAMES[b])
        axx.set_ylabel(label); axx.grid(alpha=.3); axx.legend()
    axes[-1].set_xlabel("Days")
    fig.tight_layout(); fig.savefig(out/"parameter_profiles.png",dpi=300); plt.close(fig)


def validate_external_config(cfg):
    import hashlib
    calibration_path = Path(__file__).resolve().parent.parent / "configs/calibration.json"
    a = json.loads(calibration_path.read_text(encoding="utf-8"))
    if a.get("passed") is not True or a.get("fraction") != 0.1:
        raise RuntimeError("M2 calibration not verified")
    nu = np.asarray(a["nu_per_day"], dtype=float)
    expected = 0.1 * np.asarray(a["baseline_cumulative_all_infections"]) / np.asarray(a["baseline_integral_susceptibles_person_days"])
    if not np.array_equal(nu, expected) or not np.array_equal(nu, np.array([cfg.nu1_external, cfg.nu2_external])):
        raise RuntimeError("Fixed M2 hazard differs from calibration")
    for name, digest in a["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("M2 calibration source hash changed: " + name)

def main():
    base = CONFIG
    validate_external_config(base)
    data = prepare_data(base)

    print(f"Device: {data['device']}")
    print(f"Dtype : {data['dtype']}")
    print(f"Days  : {data['n_days']}")
    print(f"COUNT_SCALE = {data['count_scale']:.6g}")
    print(f"gamma = mu = {base.gamma_fixed:.8f}")
    print(
        f"Bounds: c={base.c1_bounds}, q={base.q1_bounds}, m={base.m1_bounds}, "
        f"d={base.d1_bounds}, beta={base.beta_bounds}"
    )

    root = Path.cwd() / base.output_root
    root.mkdir(parents=True, exist_ok=True)

    best_cfg = None
    best_state = None
    best_score = np.inf
    rows = []

    if base.run_autotune:
        print("\n================ AUTOTUNE SCREENING ================")
        for idx, cand in enumerate(base.autotune_candidates, start=1):
            seed, lr, smooth, sat, beta0, cum_w, raw_w, sqrt_w, log_w = cand
            cfg = replace(
                base,
                seed=int(seed), adam_lr=float(lr),
                w_rate_smoothness=float(smooth),
                w_rate_saturation=float(sat),
                beta_init=float(beta0), w_cumulative=float(cum_w),
                w_daily_raw=float(raw_w), w_daily_sqrt=float(sqrt_w),
                w_daily_log=float(log_w),
            )
            trial_dir = root / f"_screen_trial_{idx:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = train_candidate(cfg, data, trial_dir, is_screen=True)
                val = evaluate_candidate(cfg, data, result)
                row = {
                    "trial": idx, "seed": cfg.seed, "lr": cfg.adam_lr,
                    "smooth": cfg.w_rate_smoothness,
                    "saturation": cfg.w_rate_saturation,
                    "beta_init": cfg.beta_init,
                    "beta_estimated": result["beta"],
                    "cum_weight": cfg.w_cumulative,
                    "raw_weight": cfg.w_daily_raw,
                    "sqrt_weight": cfg.w_daily_sqrt,
                    "log_weight": cfg.w_daily_log,
                    "score": val["score"],
                    "GD_R2": val["GD"]["R2"],
                    "GX_R2": val["GX"]["R2"],
                    "GD_NRMSE": val["GD"]["NRMSE"],
                    "GX_NRMSE": val["GX"]["NRMSE"],
                    "boundary_fraction": val["boundary_fraction"],
                    "roughness": val["roughness"],
                    "training_seconds": result["training_seconds"],
                }
                if val["score"] < best_score:
                    best_score = val["score"]
                    best_cfg = cfg
                    best_state = deepcopy(result["pnet_state_dict"])
            except Exception as exc:
                print(f"Trial {idx} failed: {exc}")
                row = {
                    "trial": idx, "seed": cfg.seed, "score": 1e9,
                    "GD_R2": np.nan, "GX_R2": np.nan,
                    "boundary_fraction": 1.0, "roughness": np.inf,
                }

            rows.append(row)
            print(
                f"Trial {idx}: score={row['score']:.5g}, "
                f"GD_R2={row.get('GD_R2',np.nan):.4f}, "
                f"GX_R2={row.get('GX_R2',np.nan):.4f}, "
                f"boundary={row.get('boundary_fraction',np.nan):.3f}"
            )

        screen_df = pd.DataFrame(rows).sort_values("score")
        screen_df.to_csv(root/"autotune_results.csv", index=False)
        print("\n================ SCREENING RANKING ================")
        cols = [c for c in ["trial","seed","score","GD_R2","GX_R2","boundary_fraction","roughness","beta_estimated"] if c in screen_df.columns]
        print(screen_df[cols].to_string(index=False))
        if best_cfg is None:
            raise RuntimeError("All screening trials failed.")
        final_cfg = best_cfg
    else:
        final_cfg = base
        best_state = None
        pd.DataFrame([]).to_csv(root/"autotune_results.csv", index=False)

    stamp = datetime.now().strftime("%y-%m-%d_%H%M%S")
    final_dir = root / f"Training-results-{stamp}-{base.case_name}"
    final_dir.mkdir(parents=True, exist_ok=True)

    print("\n================ FINAL WARM-STARTED FIT ================")
    final_result = train_candidate(
        final_cfg, data, final_dir, is_screen=False,
        warm_start_state_dict=best_state
    )
    val = evaluate_candidate(final_cfg, data, final_result)

    cfg_dict = asdict(final_cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, tuple):
            cfg_dict[k] = list(v)
    cfg_dict["COUNT_SCALE_used"] = data["count_scale"]
    cfg_dict["device_used"] = str(data["device"])
    cfg_dict["dtype_used"] = str(data["dtype"])
    with open(final_dir/"config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

    torch.save(
        {"parameter_net_state_dict": final_result["pnet_state_dict"],
         "config": cfg_dict, "count_scale": data["count_scale"]},
        final_dir/"parameter_net_checkpoint.pt"
    )

    save_outputs(final_cfg, data, final_result, val, final_dir)
    save_plots(data, final_result, val, final_dir)
    if (root/"autotune_results.csv").exists():
        pd.read_csv(root/"autotune_results.csv").to_csv(final_dir/"autotune_results.csv", index=False)

    fit_raw = final_result["daily_scaled"] * data["count_scale"]
    fit_gd = fit_metrics(data["y1_raw"], fit_raw[:,0])
    fit_gx = fit_metrics(data["y2_raw"], fit_raw[:,1])
    metrics_df = pd.DataFrame([
        {"method":"Differentiable ODE fit","region":"GD",**fit_gd},
        {"method":"Differentiable ODE fit","region":"GX",**fit_gx},
        {"method":"Independent saved-rate rollout","region":"GD",**val["GD"]},
        {"method":"Independent saved-rate rollout","region":"GX",**val["GX"]},
    ])
    metrics_df.to_csv(final_dir/"rollout_metrics.csv", index=False)

    with open(final_dir/"boundary_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_boundary_fraction": val["boundary_fraction"],
            "roughness": val["roughness"],
            "per_rate_boundary_fraction": val["per_rate_boundary"],
            "threshold": final_cfg.boundary_fraction_threshold,
        }, f, indent=2)

    with open(final_dir/"warnings.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val["warnings"]) if val["warnings"] else "No independent-rollout warnings.")

    summary = {
        "beta_estimated": final_result["beta"],
        "gamma_fixed": final_cfg.gamma_fixed,
        "mu_fixed": final_cfg.mu_fixed,
        "training_minutes": final_result["training_seconds"] / 60.0,
        "Differentiable_R2_GD": fit_gd["R2"],
        "Differentiable_R2_GX": fit_gx["R2"],
        "Independent_R2_GD": val["GD"]["R2"],
        "Independent_R2_GX": val["GX"]["R2"],
        "Independent_RMSE_GD": val["GD"]["RMSE"],
        "Independent_RMSE_GX": val["GX"]["RMSE"],
        "score": val["score"],
        "boundary_fraction": val["boundary_fraction"],
        "roughness": val["roughness"],
        "final_seed": final_cfg.seed,
        "final_lr": final_cfg.adam_lr,
        "final_cumulative_weight": final_cfg.w_cumulative,
        "final_daily_weights": {
            "raw": final_cfg.w_daily_raw,
            "sqrt": final_cfg.w_daily_sqrt,
            "log": final_cfg.w_daily_log,
        },
        "bounds": {
            "c": list(final_cfg.c1_bounds), "q": list(final_cfg.q1_bounds),
            "m": list(final_cfg.m1_bounds), "d": list(final_cfg.d1_bounds),
            "beta": list(final_cfg.beta_bounds),
        },
    }
    with open(final_dir/"training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ FINAL SUMMARY ================")
    print(metrics_df.to_string(index=False))
    print(f"\nbeta={final_result['beta']:.8f}, gamma=mu={final_cfg.gamma_fixed:.8f}")
    print(
        f"Boundary fraction={val['boundary_fraction']:.4f}, "
        f"roughness={val['roughness']:.4e}, score={val['score']:.6g}"
    )
    print(f"\nOutputs saved to:\n  {final_dir}")

    if val["GD"]["R2"] > 0.95 and val["GX"]["R2"] > 0.95:
        print("\nMECHANISTIC EXPORTED-PARAMETER CHECK: STRONG PASS")
    elif val["GD"]["R2"] > 0.90 and val["GX"]["R2"] > 0.90:
        print("\nMECHANISTIC EXPORTED-PARAMETER CHECK: PASS")
    else:
        print("\nMECHANISTIC EXPORTED-PARAMETER CHECK: NOT YET PASSING")


# Revision experiment settings: seed is prespecified; only the third
# originally selected training configuration is screened and warm-started.
REVISION_GROUP = "M2_external_pressure"
REVISION_SEED = 31415

def prepare_revision_run(smoke=False):
    import hashlib, platform, uuid, dataclasses
    group = Path(__file__).resolve().parent.parent
    if group.name != REVISION_GROUP or group.parent.name != "T3_model_structure":
        raise RuntimeError("Keep this script in its revision group code directory")
    project = group.parent.parent.parent
    source_config = project / "baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1/config.json"
    data_file = project / "Data/input/Data_all_sm.xlsx"
    experiment = json.loads((group / "configs/experiment.json").read_text(encoding="utf-8"))
    if experiment["seed"] != REVISION_SEED:
        raise RuntimeError("Seed differs from the pre-registered experiment")
    original = json.loads(source_config.read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(Config)}
    kw = {k: v for k, v in original.items() if k in fields}
    candidate = (REVISION_SEED, *original["autotune_candidates"][2][1:])
    parent = group / "runs" / ("pycharm_smoke" if smoke else "pycharm_manual")
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    kw.update(seed=REVISION_SEED, autotune_candidates=(candidate,), data_file=str(data_file), output_root=str(out), run_autotune=True)
    if smoke:
        kw.update(screen_adam_iterations=1, adam_iterations=1, use_lbfgs=False)
    global CONFIG
    calibration = json.loads((group / "configs/calibration.json").read_text(encoding="utf-8"))
    if calibration.get("passed") is not True or calibration.get("fraction") != 0.1: raise RuntimeError("M2 calibration not verified")
    nu_external = tuple(calibration["nu_per_day"])
    kw.update(nu1_external=nu_external[0], nu2_external=nu_external[1])
    CONFIG = Config(**kw)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {"group": REVISION_GROUP, "seed": REVISION_SEED, "smoke_only": smoke,
                "output_root": str(out), "source_config": str(source_config),
                "source_config_sha256": digest(source_config), "data_file": str(data_file),
                "data_sha256": digest(data_file), "script_sha256": digest(Path(__file__)),
                "selected_original_candidate": 3, "warm_start": "own same-seed screen weights",
                "nu_external_per_day": list(nu_external), "calibration_sha256": digest(group / "configs/calibration.json"),
                "python": platform.python_version(), "pytorch": torch.__version__,
                "run_state": "started"}
    (out / "revision_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Revision group:", REVISION_GROUP, "seed:", REVISION_SEED, "output:", out, flush=True)
    return out, metadata

def run_revision(smoke=False):
    out, metadata = prepare_revision_run(smoke=smoke)
    # Keep the normal PyCharm console while retaining a full local log.
    import sys
    class Tee:
        def __init__(self, first, second): self.first, self.second = first, second
        def write(self, text): self.first.write(text); self.second.write(text); self.second.flush()
        def flush(self): self.first.flush(); self.second.flush()
    console = (out / "training_console.log").open("w", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_stdout, console), Tee(old_stderr, console)
    try:
        main()
    except BaseException as exc:
        metadata.update(run_state="failed", error=repr(exc))
        (out / "revision_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        console.close()
    metadata["run_state"] = "completed_training_requires_independent_acceptance"
    (out / "revision_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Training completed. Independent acceptance is still required.", flush=True)

if __name__ == "__main__":
    run_revision()
