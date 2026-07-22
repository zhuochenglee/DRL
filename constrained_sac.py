"""Constrained (Lagrangian) Soft Actor-Critic.

Standard SAC optimizes the environment reward, which folds physical-constraint
violations into the objective with *fixed, hand-tuned* penalty weights. We instead
pose a constrained MDP

    max  E[ sum_t r_econ(s_t, a_t) ]
    s.t. E[ sum_t c(s_t, a_t) ] <= d   (d ~ 0)

where r_econ = -electricity_cost and c is the weight-free, unit-normalized
constraint cost exposed by the environment. The Lagrangian

    L = E[ sum_t ( r_econ - lambda * c ) ]

is optimized by an off-the-shelf SAC actor-critic on the shaped reward
(r_econ - lambda * c), while the dual variable lambda is adapted by projected
dual ascent on the constraint budget:

    lambda <- clip( lambda + eta * (E[sum_t c] - d), 0, lambda_max ).

This removes the need to hand-tune penalty weights and drives violations toward
zero automatically, which is the paper's main methodological contribution over
the vanilla penalty-reward SAC baseline.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pipeline_env import EnvConfig, PaperInspiredDynamicLinepackEnv

DEFAULT_NET_ARCH = [128, 64, 32]  # aligned with the paper's actor/critic architecture


class LagrangianRewardWrapper(gym.Wrapper):
	"""Replace the env reward by the Lagrangian r_econ - lambda * c.

	`lam` is a public attribute updated online by LagrangianDualCallback.
	"""

	def __init__(self, env, lam_init=20.0):
		super().__init__(env)
		self.lam = float(lam_init)

	def step(self, action):
		obs, _reward, terminated, truncated, info = self.env.step(action)
		shaped = info["econ_reward"] - self.lam * info["constraint_cost"]
		return obs, float(shaped), terminated, truncated, info


class LagrangianDualCallback(BaseCallback):
	"""Projected dual ascent on lambda using the per-episode constraint cost."""

	def __init__(self, wrapper, budget=0.0, dual_lr=8.0, lam_max=5.0e4, ema=0.1, verbose=0):
		super().__init__(verbose)
		self.wrapper = wrapper
		self.budget = float(budget)
		self.dual_lr = float(dual_lr)
		self.lam_max = float(lam_max)
		self.ema = float(ema)
		self._ep_cost = 0.0
		self._ep_viol = 0
		self._cost_ema = None

	def _on_step(self) -> bool:
		infos = self.locals.get("infos", [{}])
		dones = self.locals.get("dones", [False])
		for info, done in zip(infos, dones):
			if not info:
				continue
			self._ep_cost += float(info.get("constraint_cost", 0.0))
			self._ep_viol += int(info.get("violation", False))
			if done:
				# Smooth the episodic constraint cost before the dual step for stability.
				self._cost_ema = (
					self._ep_cost if self._cost_ema is None
					else (1.0 - self.ema) * self._cost_ema + self.ema * self._ep_cost
				)
				new_lam = self.wrapper.lam + self.dual_lr * (self._cost_ema - self.budget)
				self.wrapper.lam = float(np.clip(new_lam, 0.0, self.lam_max))
				self.logger.record("lagrangian/lambda", self.wrapper.lam)
				self.logger.record("lagrangian/ep_constraint_cost", self._ep_cost)
				self.logger.record("lagrangian/ep_violation_hours", self._ep_viol)
				self._ep_cost = 0.0
				self._ep_viol = 0
		return True


def train_constrained_sac(
	config: EnvConfig,
	total_timesteps: int,
	seed: int = 0,
	lam_init: float = 20.0,
	dual_lr: float = 8.0,
	lam_max: float = 5.0e4,
	budget: float = 0.0,
	learning_rate: float = 1e-3,
	gamma: float = 0.995,
	net_arch=None,
	normalize_reward: bool = True,
	tensorboard_log=None,
	tb_log_name: str = "csac",
	extra_callbacks=None,
):
	"""Train Constrained-SAC; returns (model, wrapper)."""
	net_arch = list(net_arch) if net_arch is not None else list(DEFAULT_NET_ARCH)
	base_env = PaperInspiredDynamicLinepackEnv(config=config)
	wrapper = LagrangianRewardWrapper(base_env, lam_init=lam_init)
	train_env = DummyVecEnv([lambda: wrapper])
	if normalize_reward:
		# Normalize the (wide-range, λ-scaled) return so the critic trains stably.
		train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=gamma)

	model = SAC(
		"MlpPolicy", train_env,
		verbose=0,
		seed=seed,
		learning_rate=learning_rate,
		gamma=gamma,
		policy_kwargs=dict(net_arch=net_arch),
		tensorboard_log=tensorboard_log,
	)
	dual_cb = LagrangianDualCallback(wrapper, budget=budget, dual_lr=dual_lr, lam_max=lam_max)
	callbacks = [dual_cb] + (list(extra_callbacks) if extra_callbacks else [])
	model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=tb_log_name)
	return model, wrapper


if __name__ == "__main__":
	from baselines import evaluate_model

	cfg = EnvConfig()
	model, wrapper = train_constrained_sac(cfg, total_timesteps=20000, seed=0)
	eval_env = PaperInspiredDynamicLinepackEnv(config=cfg)
	m = evaluate_model(eval_env, model)
	print(f"Constrained-SAC | final lambda={wrapper.lam:.1f} | cost={m['total_cost']:.2f} "
	      f"| violation_hours={m['violation_hours']} | mean_eff={m['mean_eff']:.3f}")
