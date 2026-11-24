# Command-refusal modeling for 110 kV open-ring restoration

This repository contains the core and timing experiments for Dongyue Kan’s manuscript *Decision value of command-refusal modeling for sequential restoration of 110 kV open-ring networks*. It evaluates when modeling a possible first tie-close refusal changes a restoration policy’s decisions and expected energy not supplied (ENS).

The experiment compares seven policies on synthetic 6-, 8-, and 10-station open rings. Each policy executes both success and refusal outcomes for each background. The core suite has 144 backgrounds and 2,016 executions. The timing suite has 72 backgrounds and 1,008 executions. The specified probabilities are scenario inputs. All policies share an admissibility gate, and a second AC implementation checks the balanced steady-state calculations. The model is for research simulation. Utility operation and relay coordination require separate engineering validation.

## Quick start

Use Python 3.12 and run these commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest tests -q
python src/ee_run.py --suite core --limit 1 --output results/smoke
```

The smoke run should produce 14 executions for one background, along with `metrics.csv`, `manifest.json`, and `independent_ac_checks.json` under `results/smoke/`.

## Reproduce the core and timing experiments

```bash
python src/ee_run.py --suite core
python src/ee_run.py --suite stress
python src/ee_analyze.py
python src/ee_plot.py
python src/ee_diagnostics_external.py
```

These commands write under `results/`. Use the frozen cases under `data/ee/` for comparisons with the reference results. `python src/ee_generate.py` regenerates the synthetic case inputs from their seeded design and writes a new protocol timestamp.

| Path | Contents |
| --- | --- |
| `src/restoration.py` | State transitions, command times, admissibility gate, and baseline policies |
| `src/contingent_planner.py` | Finite-state planning with refusal and rearm recourse |
| `src/ac_model.py`, `src/independent_ac.py` | Primary pandapower calculations and independent rectangular AC checks |
| `data/ee/` | Frozen synthetic networks, case inputs, and experiment protocol |
| `reference-results/` | Original numerical summaries and per-execution metrics for the core and timing suites |
| `tests/` | Electrical, policy, and state-transition checks |

At the specified evaluation probability of 0.25, the matched contingent policy lowers mean ENS by about 0.99 kWh per background relative to nominal sequence search in the original core run. The reference tables retain the full probability grid, including unfavorable comparisons. Timing varies by environment.

## Data provenance

Network geometry, loads, source ceilings, timings, and events are synthetic. Selected line and transformer electrical constants are derived from the [pandapower standard-type documentation](https://pandapower.readthedocs.io/en/latest/std_types/basic.html). `data/provenance.json` records the types, modeling simplifications, and generation seed. The derived inputs use their own synthetic network layout and operating assumptions.
