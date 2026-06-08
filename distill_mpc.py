"""Distill the MPC controller into a millisecond feedback policy.

The MPC here is the certainty-equivalent DP feedback law built on the nominal
forecast and executed closed-loop. It is feasible and near-optimal but costs
~1 s per dispatch to build/run. We clone it into a small MLP by behavioral
cloning over states visited on many perturbed scenarios, optionally refined with
one DAgger round (relabel the student's own visited states with the MPC action).
The result acts in milliseconds while matching MPC cost and feasibility.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv
from DP import build_dp_policy
from baselines import evaluate_model, run_mpc

NET_ARCH = [128, 64, 32]


# ── teacher: nominal-DP feedback (= MPC) action lookup ────────────────────────
class MPCTeacher:
	def __init__(self, env, nominal_d, nominal_p, n_p=121, n_a=101):
		self.p_grid, self.a_grid, self.best_action = build_dp_policy(
			env, nominal_d, nominal_p, n_p=n_p, n_a=n_a)
		self.dp_val = float(self.p_grid[1] - self.p_grid[0])
		self.n_p = len(self.p_grid)

	def act(self, env, hour):
		p2, p3 = float(env.P_internal[0]), float(env.P_internal[1])
		i2 = int(np.clip(round((p2 - self.p_grid[0]) / self.dp_val), 0, self.n_p - 1))
		i3 = int(np.clip(round((p3 - self.p_grid[0]) / self.dp_val), 0, self.n_p - 1))
		return float(self.a_grid[int(self.best_action[hour, i2, i3])])


# ── student MLP ───────────────────────────────────────────────────────────────
class MLP(nn.Module):
	def __init__(self, in_dim, arch=NET_ARCH):
		super().__init__()
		layers, d = [], in_dim
		for h in arch:
			layers += [nn.Linear(d, h), nn.ReLU()]
			d = h
		layers += [nn.Linear(d, 1)]
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class DistilledPolicy:
	"""Wraps the trained MLP with an SB3-style predict() so the shared evaluator works."""

	def __init__(self, model, lo=1.0, hi=2.0):
		self.model = model.eval()
		self.lo, self.hi = lo, hi

	def predict(self, obs, deterministic=True):
		with torch.no_grad():
			a = self.model(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)).item()
		return np.array([float(np.clip(a, self.lo, self.hi))], dtype=np.float32), None


# ── data collection (teacher rollouts) ────────────────────────────────────────
def _rollout_collect(env, teacher, demand, prices, student=None, p_explore=0.0):
	"""Roll out on a scenario; return (obs, teacher_action) pairs. If a student is
	given, it drives the trajectory (DAgger) so the teacher labels visited states."""
	options = {"demand": np.asarray(demand), "prices": np.asarray(prices)}
	obs, _ = env.reset(options=options)
	X, Y = [], []
	for hour in range(env.horizon):
		a_teacher = teacher.act(env, hour)
		X.append(np.asarray(obs, dtype=np.float32)); Y.append([a_teacher])
		if student is not None and np.random.rand() > p_explore:
			a_use, _ = student.predict(obs); a_use = float(a_use[0])
		else:
			a_use = a_teacher
		obs, _, term, trunc, _ = env.step(np.array([a_use], dtype=np.float32))
		if term or trunc:
			break
	return X, Y


def collect_dataset(env, teacher, scenarios, student=None, p_explore=0.0):
	X, Y = [], []
	for d, p in scenarios:
		xs, ys = _rollout_collect(env, teacher, d, p, student=student, p_explore=p_explore)
		X += xs; Y += ys
	return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def train_bc(X, Y, in_dim, epochs=300, lr=1e-3, seed=0):
	torch.manual_seed(seed)
	model = MLP(in_dim)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	lossf = nn.MSELoss()
	Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
	for _ in range(epochs):
		opt.zero_grad()
		loss = lossf(model(Xt), Yt)
		loss.backward(); opt.step()
	return model


def make_scenarios(env, n, noise_scale, seed):
	rng = np.random.default_rng(seed)
	bd = np.asarray(env.demand_series, float); bp = np.asarray(env.tou_price_series, float)
	out = []
	for _ in range(n):
		d = np.clip(bd * (1.0 + rng.normal(0, noise_scale, bd.shape)), 50, 400)
		out.append((d, np.roll(bp, int(rng.integers(-2, 3)))))
	return out


def distill(cfg=None, n_train=200, n_dagger=1, epochs=300, seed=0):
	cfg = cfg or EnvConfig(noise_scale=0.0)
	env = PaperInspiredDynamicLinepackEnv(config=cfg)
	nd, npp = np.asarray(env.demand_series, float), np.asarray(env.tou_price_series, float)
	teacher = MPCTeacher(env, nd, npp)
	in_dim = env.observation_space.shape[0]

	scen = [(nd, npp)] + make_scenarios(env, n_train, 0.10, seed)
	X, Y = collect_dataset(env, teacher, scen)
	model = train_bc(X, Y, in_dim, epochs=epochs, seed=seed)
	student = DistilledPolicy(model)

	# DAgger refinement: relabel states the student actually visits
	for _ in range(n_dagger):
		Xd, Yd = collect_dataset(env, teacher, scen, student=student, p_explore=0.1)
		X = np.concatenate([X, Xd]); Y = np.concatenate([Y, Yd])
		model = train_bc(X, Y, in_dim, epochs=epochs, seed=seed)
		student = DistilledPolicy(model)
	return student, teacher


if __name__ == "__main__":
	cfg = EnvConfig(noise_scale=0.0)
	ev = PaperInspiredDynamicLinepackEnv(config=cfg)
	nd, npp = np.asarray(ev.demand_series, float), np.asarray(ev.tou_price_series, float)

	t = time.time(); student, _ = distill(cfg); t_train = time.time() - t

	# nominal
	t = time.time(); m = evaluate_model(ev, student, nd, npp); infer = time.time() - t
	mpc = run_mpc(ev, nd, npp, nd, npp, n_p=121, n_a=101)
	print(f"MPC teacher      cost={mpc['total_cost']:6.2f} viol={mpc['violation_hours']:2d}")
	print(f"Distilled-MPC    cost={m['total_cost']:6.2f} viol={m['violation_hours']:2d} "
	      f"eff={m['mean_eff']:.3f} tlp={m['terminal_linepack_gap']:.0f}  "
	      f"infer={infer*1000:.1f}ms (train {t_train:.0f}s)")

	# robustness
	scen = make_scenarios(ev, 6, 0.10, seed=123)
	sc = [evaluate_model(ev, student, d, p)["total_cost"] for d, p in scen]
	sv = [evaluate_model(ev, student, d, p)["violation_hours"] for d, p in scen]
	print(f"Distilled-MPC robust: cost={np.mean(sc):.2f}±{np.std(sc):.1f} viol={np.mean(sv):.1f}")
