from src.modal_training_plan import (
    TrialResult,
    choose_best_trial,
    estimate_hourly_cost_usd,
)


def test_choose_best_trial_prefers_highest_throughput_under_vram_limit():
    trials = [
        TrialResult(batch_size=4, grad_accum=8, optimizer_steps=30, samples_per_sec=22.0, peak_vram_gib=10.0, oom=False),
        TrialResult(batch_size=8, grad_accum=4, optimizer_steps=30, samples_per_sec=28.5, peak_vram_gib=18.0, oom=False),
        TrialResult(batch_size=16, grad_accum=2, optimizer_steps=30, samples_per_sec=31.0, peak_vram_gib=23.5, oom=False),
    ]
    # With 24GB card and 95% limit => max 22.8GiB, so 16x2 should be excluded.
    best = choose_best_trial(trials, total_vram_gib=24.0, max_vram_fraction=0.95)
    assert best is not None
    assert best.batch_size == 8
    assert best.grad_accum == 4


def test_choose_best_trial_ignores_oom_trials():
    trials = [
        TrialResult(batch_size=8, grad_accum=4, optimizer_steps=30, samples_per_sec=25.0, peak_vram_gib=19.0, oom=False),
        TrialResult(batch_size=16, grad_accum=2, optimizer_steps=5, samples_per_sec=40.0, peak_vram_gib=0.0, oom=True),
    ]
    best = choose_best_trial(trials, total_vram_gib=24.0)
    assert best is not None
    assert best.batch_size == 8


def test_choose_best_trial_fallback_to_non_oom_if_all_over_vram_target():
    trials = [
        TrialResult(batch_size=8, grad_accum=4, optimizer_steps=30, samples_per_sec=20.0, peak_vram_gib=23.7, oom=False),
        TrialResult(batch_size=4, grad_accum=8, optimizer_steps=30, samples_per_sec=18.0, peak_vram_gib=23.4, oom=False),
    ]
    # Both exceed 95% of 24GB (22.8), should still return best non-OOM.
    best = choose_best_trial(trials, total_vram_gib=24.0, max_vram_fraction=0.95)
    assert best is not None
    assert best.batch_size == 8


def test_estimate_hourly_cost_usd_matches_modal_formula():
    # A10 base + 4 CPU + 8 GiB memory
    cost = estimate_hourly_cost_usd(gpu_rate_per_sec=0.000306, cpu_cores=4.0, memory_gib=8.0)
    assert abs(cost - 1.354) < 0.02
