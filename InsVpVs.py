"""Estimate local, in-situ Vp/Vs from similar earthquake clusters.

Based on Lin & Shearer (2007), BSSA 97(2), 379-388, doi:10.1785/0120060115.

For each earthquake pair, pair P and S differential times at the SAME station,
robustly demean P and S separately, and fit a common slope to the demeaned
points using iterative, rescaled, robust orthogonal-distance regression.

Input formats:
  FDTCC or hypoDD_cc:  # event_id_1 event_id_2
                        STATION  differential_time  correlation_or_weight  P|S
  hypoDD_ct:             # event_id_1 event_id_2
                        STATION  time_event_1 time_event_2 weight P|S

IMPORTANT:
* Both P and S differential times must use the SAME event-order sign.
* For hypoDD_ct the fourth column is a pick weight, NOT a CC coefficient.
* Check that FDTCC's time field represents comparable differential arrival
  times, not uncorrected residual shifts with different reference times.
* Inferred Vp/Vs is a cluster-scale near-source estimate; limited geometry,
  anisotropy, and P/S ray-path differences can produce systematic bias.

Dependencies: numpy, scipy, matplotlib. Run: python InsVpVs_revised.py
Edit the CONFIGURATION block to match your input files.
"""
from __future__ import annotations

import csv
import glob
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a display.
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import linregress

# ============================= CONFIGURATION =============================
INPUT_GLOB = "*.cc"                # e.g., "*cc", "*.cc", "data/*.ct"
INPUT_FORMAT = "FDTCC"             # FDTCC | hypoDD_cc | hypoDD_ct
OUTPUT_DIR = Path("vpvs_results")

MIN_CC = 0.80                     # Only for FDTCC/hypoDD_cc; no CC interpretation for dt.ct.
MIN_CT_WEIGHT = None              # Only for hypoDD_ct; None means no weight filter.
MAX_ABS_DT_P = None               # seconds; None disables absolute P cutoff.
MAX_ABS_DT_S = None               # seconds; None disables absolute S cutoff.
# Different equal-absolute P/S cutoffs can censor S differentially and bias R.

MIN_COMMON_STATIONS = 5            # Minimum matched P/S stations PER event pair.
MIN_TOTAL_OBSERVATIONS = 50        # Matched station observations across cluster.
MIN_EVENT_PAIRS = 5                # Independent distinct pair records per cluster.

INITIAL_RATIO = 1.732
RATIO_BOUNDS = (1.0, 4.0)          # Physical search interval; inspect if solution hits a bound.
MAX_ITER = 30
REL_TOL = 1e-6
THROUGH_ORIGIN = False             # False fits an extra global intercept (diagnostic).

HUBER_DELTA = None                # seconds, on S/R-scaled perpendicular residuals.
HUBER_K = 1.345                   # Adaptive threshold = K * 1.4826 * residual MAD.
MIN_HUBER_DELTA = 0.0005          # seconds, prevents numerical zero thresholds.

N_BOOTSTRAP = 300                 # Resample whole earthquake pairs. Set 0 to disable.
RANDOM_SEED = 2026
SHOW_LEAST_SQUARES = True
RUN_STATION_JACKKNIFE = True        # Diagnostic: refit after omitting each station.
PLOT_DPI = 400
# ========================================================================


@dataclass
class EventPair:
    id1: int
    id2: int
    stations: tuple[str, ...]
    p: np.ndarray                 # Already robustly demeaned for this pair.
    s: np.ndarray
    quality: np.ndarray           # Average of P/S CC or weights; display only.


@dataclass
class FitResult:
    ratio: float
    intercept: float              # In original S-time units (seconds).
    iterations: int
    converged: bool
    huber_delta: float             # In scaled perpendicular-distance units (seconds).
    at_bound: bool


def huber_loss_sum(residuals: np.ndarray, delta: float) -> float:
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be finite and positive")
    a = np.abs(residuals)
    return float(np.sum(np.where(a <= delta, 0.5 * a*a, delta*(a - 0.5*delta))))


def robust_mean(values: np.ndarray, delta: float) -> float:
    """Exact Huber M-estimator of location using its score equation."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Need non-empty, finite data to calculate robust mean")
    lower, upper = float(np.min(values)), float(np.max(values))
    if lower == upper:
        return lower
    # derivative wrt location mu is minus sum(clip(values - mu, -delta, delta))
    def score(mu):
        return float(np.sum(np.clip(values - mu, -delta, delta)))
    return float(brentq(score, lower, upper, xtol=1e-12))


def robust_scale(residuals: np.ndarray, floor: float) -> float:
    """MAD-based Gaussian-equivalent residual standard deviation."""
    center = float(np.median(residuals))
    return max(1.4826 * float(np.median(np.abs(residuals - center))), floor)


def parse_input(filename: Path) -> tuple[list[EventPair], dict[str, int]]:
    """Read event-pair blocks, match station/phase and demean every block."""
    fmt = INPUT_FORMAT.lower()
    if fmt not in {"fdtcc", "hypodd_cc", "hypodd_ct"}:
        raise ValueError("INPUT_FORMAT must be FDTCC, hypoDD_cc, or hypoDD_ct")
    if MIN_COMMON_STATIONS < 2:
        raise ValueError("MIN_COMMON_STATIONS must be at least 2")

    stats = dict(headers=0, rejected_pairs=0, rejected_rows=0,
                 duplicates=0, orphan_phase_rows=0)
    pairs = []
    seen_pairs = set()
    current_ids = None
    p_obs: dict[str, tuple[float, float]] = {}
    s_obs: dict[str, tuple[float, float]] = {}

    def flush_pair():
        nonlocal p_obs, s_obs, current_ids
        if current_ids is None:
            return
        common = sorted(p_obs.keys() & s_obs.keys())
        stats["orphan_phase_rows"] += (len(p_obs) - len(common)) + (len(s_obs) - len(common))
        if len(common) < MIN_COMMON_STATIONS:
            stats["rejected_pairs"] += 1
            return
        p = np.array([p_obs[st][0] for st in common], dtype=float)
        s = np.array([s_obs[st][0] for st in common], dtype=float)
        quality = np.array([(p_obs[st][1] + s_obs[st][1])/2 for st in common])
        # Robust location is used for BOTH phases, never just for P or last pair.
        # A physically meaningful slope requires within-pair station variation.
        p -= robust_mean(p, max(MIN_HUBER_DELTA, pair_centering_delta(p)))
        s -= robust_mean(s, max(MIN_HUBER_DELTA, pair_centering_delta(s)))
        if np.ptp(p) < 1e-10:
            stats["rejected_pairs"] += 1
            return
        pairs.append(EventPair(*current_ids, tuple(common), p, s, quality))

    with filename.open(encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                fields = line[1:].split()
                if len(fields) < 2:
                    continue
                try:
                    ids = int(fields[0]), int(fields[1])
                except ValueError:
                    # This may be a comment rather than a pair header.
                    continue
                flush_pair()
                pair_key = tuple(sorted(ids))
                if pair_key in seen_pairs:
                    raise ValueError(
                        f"{filename}:{line_num}: repeated event-pair IDs {ids}. "
                        "Remove duplicate/reversed pair blocks before fitting."
                    )
                seen_pairs.add(pair_key)
                current_ids = ids
                p_obs, s_obs = {}, {}
                stats["headers"] += 1
                continue
            if current_ids is None:
                raise ValueError(f"{filename}:{line_num}: data before first '# ID1 ID2' header")
            fields = line.split()
            expected = 5 if fmt == "hypodd_ct" else 4
            if len(fields) != expected:
                raise ValueError(
                    f"{filename}:{line_num}: expected {expected} fields for "
                    f"{INPUT_FORMAT}, found {len(fields)}: {line}"
                )
            try:
                if fmt == "hypodd_ct":
                    station, t1, t2, weight, phase = fields
                    diff = float(t1) - float(t2)
                    q = float(weight)
                else:
                    station, diff_str, corr, phase = fields
                    diff, q = float(diff_str), float(corr)
            except ValueError as exc:
                raise ValueError(f"{filename}:{line_num}: invalid numeric field: {line}") from exc
            phase = phase.upper()
            if phase not in {"P", "S"} or not np.isfinite(diff) or not np.isfinite(q):
                stats["rejected_rows"] += 1
                continue
            if fmt == "hypodd_ct":
                if MIN_CT_WEIGHT is not None and q < MIN_CT_WEIGHT:
                    stats["rejected_rows"] += 1
                    continue
            elif q < MIN_CC:
                stats["rejected_rows"] += 1
                continue
            dt_limit = MAX_ABS_DT_P if phase == "P" else MAX_ABS_DT_S
            if dt_limit is not None and abs(diff) > dt_limit:
                stats["rejected_rows"] += 1
                continue
            phase_dict = p_obs if phase == "P" else s_obs
            if station in phase_dict:
                stats["duplicates"] += 1
                # Keep the highest reported CC/weight for a duplicate entry.
                if q <= phase_dict[station][1]:
                    continue
            phase_dict[station] = (diff, q)
    flush_pair()  # IMPORTANT: exactly the same processing for the final pair.
    return pairs, stats


def pair_centering_delta(values: np.ndarray) -> float:
    """Threshold for within-pair robust demeaning; different from fit delta."""
    return HUBER_K * robust_scale(values, MIN_HUBER_DELTA)


def flatten(pairs: list[EventPair]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not pairs:
        raise ValueError("No valid event pairs")
    return (np.concatenate([pair.p for pair in pairs]),
            np.concatenate([pair.s for pair in pairs]),
            np.concatenate([pair.quality for pair in pairs]))


def orthogonal_huber_fit(x: np.ndarray, y: np.ndarray, delta: float,
                          slope_bounds: tuple[float, float],
                          through_origin: bool = False) -> tuple[float, float]:
    """Minimize Huber losses on orthogonal (not vertical) point-to-line distances."""
    lo, hi = slope_bounds
    if not 0 < lo < hi:
        raise ValueError("Invalid positive slope search bounds")

    def intercept_for(m: float) -> float:
        if through_origin:
            return 0.0
        # Perpendicular Huber delta corresponds to vertical delta * sqrt(1+m^2).
        return robust_mean(y - m*x, delta * np.hypot(1.0, m))

    def objective(m: float) -> float:
        b = intercept_for(m)
        residual = (y - m*x - b)/np.hypot(1.0, m)
        return huber_loss_sum(residual, delta)

    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-10, "maxiter": 500})
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Robust orthogonal slope minimization failed")
    m = float(result.x)
    return m, intercept_for(m)


def fit_vpvs(p: np.ndarray, s: np.ndarray, *, verbose: bool = False) -> FitResult:
    """Iteratively rescale S, fit robust orthogonal line, update Vp/Vs."""
    if p.shape != s.shape or len(p) < 3:
        raise ValueError("P and S must have equal lengths >= 3")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("Non-finite differential times")
    if np.std(p) < 1e-10:
        raise ValueError("P differential-time spread too small to estimate slope")
    rlo, rhi = RATIO_BOUNDS
    if not 0 < rlo < INITIAL_RATIO < rhi:
        raise ValueError("INITIAL_RATIO must lie inside RATIO_BOUNDS")

    R = INITIAL_RATIO
    intercept = 0.0
    converged = False
    delta = None
    for iteration in range(1, MAX_ITER + 1):
        y = s/R
        if HUBER_DELTA is None:
            # Estimate scatter in equal-unit (scaled) P/S space. Refine in later
            # iterations as the estimated slope approaches one.
            residual_guess = (y - p)/np.sqrt(2.0)
            delta = HUBER_K * robust_scale(residual_guess, MIN_HUBER_DELTA)
        else:
            delta = float(HUBER_DELTA)
        slope, scaled_intercept = orthogonal_huber_fit(
            p, y, delta, (rlo/R, rhi/R), THROUGH_ORIGIN)
        next_R = R*slope
        intercept = R*scaled_intercept
        if verbose:
            print(f"    iteration {iteration:2}: R={next_R:.6f}, "
                  f"delta={delta:.6g} s, intercept={intercept:.6g} s")
        if abs(next_R - R) / R < REL_TOL:
            converged = True
            R = next_R
            break
        R = next_R
    if not converged:
        warnings.warn("Robust Vp/Vs iteration did not converge", RuntimeWarning)
    at_bound = ((R - rlo) < 1e-3) or ((rhi - R) < 1e-3)
    return FitResult(float(R), float(intercept), iteration,
                     converged, float(delta), at_bound)


def bootstrap_vpvs(pairs: list[EventPair], n_bootstrap: int,
                   seed: int) -> tuple[float, float, float, int]:
    """Cluster bootstrap by event PAIR, keeping all its stations together.

    Note: pairs sharing an event are still dependent, so bootstrap spread can
    underestimate true uncertainty. Systematic ray geometry bias is excluded.
    """
    if n_bootstrap <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    n = len(pairs)
    ratios = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sampled = [pairs[i] for i in indices]
        p, s, _ = flatten(sampled)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fit = fit_vpvs(p, s)
            if fit.converged and not fit.at_bound:
                ratios.append(fit.ratio)
        except (ValueError, RuntimeError, FloatingPointError):
            pass
    if len(ratios) < max(20, int(n_bootstrap*0.8)):
        warnings.warn(f"Only {len(ratios)}/{n_bootstrap} bootstrap fits succeeded")
    if len(ratios) < 2:
        return float("nan"), float("nan"), float("nan"), len(ratios)
    a = np.asarray(ratios)
    return float(np.std(a, ddof=1)), *map(float, np.percentile(a, [2.5, 97.5])), len(a)



def leave_one_station_out(pairs: list[EventPair]) -> list[dict[str, object]]:
    """Sensitivity diagnostic. Fully re-demean pairs after dropping one station.

    Unlike a pair bootstrap, this is a geometry-influence check, not a formal
    confidence interval. Do not compare uncentered subsets of event pairs.
    """
    stations = sorted({sta for pair in pairs for sta in pair.stations})
    result = []
    for omitted in stations:
        new_pairs = []
        for pair in pairs:
            keep = np.array([sta != omitted for sta in pair.stations], dtype=bool)
            if int(np.count_nonzero(keep)) < MIN_COMMON_STATIONS:
                continue
            p = pair.p[keep].copy()
            s = pair.s[keep].copy()
            p -= robust_mean(p, max(MIN_HUBER_DELTA, pair_centering_delta(p)))
            s -= robust_mean(s, max(MIN_HUBER_DELTA, pair_centering_delta(s)))
            if np.ptp(p) < 1e-10:
                continue
            new_pairs.append(EventPair(pair.id1, pair.id2,
                                       tuple(sta for sta, k in zip(pair.stations, keep) if k),
                                       p, s, pair.quality[keep]))
        nobs = sum(len(pair.p) for pair in new_pairs)
        row = dict(station_omitted=omitted, n_pairs=len(new_pairs),
                   n_measurements=nobs, vpvs=float("nan"))
        if len(new_pairs) >= MIN_EVENT_PAIRS and nobs >= MIN_TOTAL_OBSERVATIONS:
            p, s, _ = flatten(new_pairs)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fit = fit_vpvs(p, s)
                if fit.converged and not fit.at_bound:
                    row["vpvs"] = fit.ratio
            except (ValueError, RuntimeError):
                pass
        result.append(row)
    return result


def plot_fit(p: np.ndarray, s: np.ndarray, quality: np.ndarray,
             fit: FitResult, output: Path, label: str, n_pairs: int,
             uncertainty: tuple[float, float, float, int]) -> None:
    se, lower, upper, n_boot = uncertainty
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(p, s, c=quality, cmap="viridis", s=11, alpha=0.7,
                    linewidths=0, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean P/S CC" if INPUT_FORMAT.lower() != "hypodd_ct" else "Mean P/S weight")
    grid = np.linspace(float(p.min()), float(p.max()), 200)
    suffix = f" ± {se:.3f}" if np.isfinite(se) else ""
    ax.plot(grid, fit.ratio*grid + fit.intercept, color="crimson", lw=2,
            label=f"Robust TLS Vp/Vs = {fit.ratio:.3f}{suffix}")
    if SHOW_LEAST_SQUARES:
        ls = linregress(p, s)
        ax.plot(grid, ls.slope*grid + ls.intercept, "--", color="royalblue",
                label=f"Ordinary LS slope = {ls.slope:.3f}")
    ax.set(xlabel="Demeaned differential P-arrival time (s)",
           ylabel="Demeaned differential S-arrival time (s)")
    note = f"{n_pairs:,} event pairs; {len(p):,} P/S station matches"
    if np.isfinite(lower):
        note += f"\nPair bootstrap 95% CI [{lower:.3f}, {upper:.3f}] ({n_boot} runs)"
    ax.set_title(f"{label}\n{note}")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def process_file(filename: Path, output_dir: Path) -> dict[str, object] | None:
    pairs, stats = parse_input(filename)
    p = np.array([], dtype=float)
    if pairs:
        p, s, quality = flatten(pairs)
    print(f"  Headers={stats['headers']}; valid pairs={len(pairs)}; "
          f"matched measurements={len(p)}; rejected pairs={stats['rejected_pairs']}; "
          f"rejected rows={stats['rejected_rows']}; duplicates={stats['duplicates']}")
    if len(pairs) < MIN_EVENT_PAIRS or len(p) < MIN_TOTAL_OBSERVATIONS:
        print(f"  Skipped: require >= {MIN_EVENT_PAIRS} pairs, "
              f">= {MIN_TOTAL_OBSERVATIONS} matched measurements")
        return None

    fit = fit_vpvs(p, s, verbose=True)
    if fit.at_bound:
        warnings.warn(f"{filename.name}: fitted Vp/Vs near RATIO_BOUNDS; inspect data")
    uncertainty = bootstrap_vpvs(pairs, N_BOOTSTRAP, RANDOM_SEED)
    se, low, high, nboot = uncertainty
    print(f"  Vp/Vs={fit.ratio:.5f}, 95% CI=[{low:.5f}, {high:.5f}], "
          f"bootstrap std={se:.5f}, converged={fit.converged}")
    plot_fit(p, s, quality, fit, output_dir / f"{filename.stem}_vpvs.png",
             filename.name, len(pairs), uncertainty)
    station_sensitivity = []
    if RUN_STATION_JACKKNIFE:
        station_sensitivity = leave_one_station_out(pairs)
        with (output_dir / f"{filename.stem}_station_jackknife.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["station_omitted", "n_pairs",
                                                        "n_measurements", "vpvs"])
            writer.writeheader()
            writer.writerows(station_sensitivity)
        valid = np.asarray([row["vpvs"] for row in station_sensitivity
                            if np.isfinite(row["vpvs"])])
        if valid.size:
            print(f"  Station omission range: {valid.min():.5f} to {valid.max():.5f} "
                  f"({valid.size} valid exclusions)")

    return dict(file=str(filename), vpvs=fit.ratio, intercept_s=fit.intercept,
                bootstrap_std=se, ci95_low=low, ci95_high=high,
                bootstrap_success=nboot, n_pairs=len(pairs),
                n_measurements=len(p), n_stations=len(set(st for pair in pairs for st in pair.stations)),
                n_rejected_pairs=stats["rejected_pairs"],
                n_rejected_rows=stats["rejected_rows"],
                n_duplicate_station_phases=stats["duplicates"],
                n_orphan_phase_rows=stats["orphan_phase_rows"],
                station_jackknife_valid=sum(np.isfinite(row["vpvs"])
                                            for row in station_sensitivity),
                station_jackknife_min=min(
                    [row["vpvs"] for row in station_sensitivity
                     if np.isfinite(row["vpvs"])] or [float("nan")]),
                station_jackknife_max=max(
                    [row["vpvs"] for row in station_sensitivity
                     if np.isfinite(row["vpvs"])] or [float("nan")]),
                fit_iterations=fit.iterations, converged=fit.converged,
                at_ratio_bound=fit.at_bound, huber_delta_s=fit.huber_delta)


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    files = [Path(name) for name in sorted(glob.glob(INPUT_GLOB))]
    if not files:
        print(f"No input files match {INPUT_GLOB!r} in {Path.cwd()}")
        return
    if N_BOOTSTRAP < 0:
        raise ValueError("N_BOOTSTRAP must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for filename in files:
        print(f"\nProcessing {filename} ({INPUT_FORMAT})")
        try:
            row = process_file(filename, output_dir)
            if row:
                results.append(row)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"  ERROR: {exc}")
    if results:
        output = output_dir / "vpvs_summary.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {len(results)} results to {output}")
    else:
        print("\nNo valid results; check input format and selection thresholds.")


if __name__ == "__main__":
    main()
