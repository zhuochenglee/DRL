# Training Visualization Plan

Use Stable-Baselines3's native TensorBoard integration instead of introducing a separate tracking stack. The recommended path is to enable SB3 logging first, then add compressor-specific metrics through a callback so the charts reflect both RL optimization and physical safety behavior.

## Steps

1. Enable native TensorBoard logging in `envrionment_2.py` by passing a log directory to `SAC(...)`. This is the minimum change required to produce training curves.
2. Add an explicit runtime dependency for TensorBoard in `pyproject.toml` so the visualization tool is available in the project environment.
3. Fix reproducibility before comparing runs by replacing the remaining global RNG usage in `SimpleCompressorEnv.reset()` with `self.np_random`, and set a fixed `seed` in `SAC(...)` for training runs. This should be done before analyzing charts across experiments.
4. Add environment-specific metric logging via a Stable-Baselines3 callback in `envrionment_2.py`. Record `p1`, `p2`, `power`, and `violation` from the `info` dict to TensorBoard on each step.
5. Optionally wrap the environment with `Monitor` in `envrionment_2.py` to persist episode statistics and selected `info_keywords` alongside TensorBoard logs.
6. Run training and inspect TensorBoard to confirm the default SB3 metrics (`rollout/*`, `train/*`, `time/*`) and custom compressor metrics appear as expected.
7. If charts are hard to interpret, revisit the environment design: the current one-step episode structure means episode metrics are shallow, so longer episodes may be needed later for more informative rollout charts. This is out of scope for the initial visualization integration.

## Relevant Files

- `envrionment_2.py`: training entry point, `SimpleCompressorEnv.reset()`, `SimpleCompressorEnv.step()`, `SAC(...)` construction, and potential callback or `Monitor` integration.
- `pyproject.toml`: add `tensorboard` as an explicit dependency.

## Verification

1. Install dependencies from `pyproject.toml` and verify `tensorboard --version` works in the project environment.
2. Run the training script and confirm a TensorBoard log directory is created under the configured path.
3. Start TensorBoard with the configured log directory and verify default SB3 charts appear.
4. Confirm custom charts for `env/p1`, `env/p2`, `env/power`, and `env/violation` update during training.
5. Run the script twice with the same seed and verify the initial sampled demand and price sequence is reproducible enough for comparison.

## Decisions

- Use TensorBoard because Stable-Baselines3 supports it natively; avoid introducing a heavier experiment platform unless multi-run comparison, artifact storage, or remote dashboards become necessary.
- Include custom physical-process metrics in the first logging pass; default RL losses alone are not enough to judge control quality in this environment.
- Exclude episode-structure redesign from the first iteration; fix observability first, then decide whether the single-step MDP needs reformulation.

## Further Considerations

1. If you want richer experiment management later, consider Weights & Biases as a second-stage upgrade, but it is not necessary for this code to get useful training plots.
2. The current reward penalty magnitude (`1e6`) may dominate TensorBoard scales; if plots are unreadable, split safety metrics into separate charts instead of changing reward design immediately.
3. The remaining RNG call for `price_state` should be migrated to `self.np_random` before relying on chart-to-chart comparisons across seeds.