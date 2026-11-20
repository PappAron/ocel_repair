# OCEL repair: reconstructing missing event-to-object relations

Code for the paper *Object-Centric Event Log Repair Through Graph Neural Networks*
(Áron Balázs Papp, István Koren; ICPM 2027, submission 127, revised version).

An event that has lost a reference to one of its objects is repaired by ranking all objects of the log
by how likely they are to be the missing one (link prediction on the heterogeneous event–object graph).
Our model combines a heterogeneous GraphSAGE encoder, which represents objects by their type and degree
instead of per-object parameters, with pairwise evidence: object co-occurrences, object-to-object (O2O)
relations and temporal proximity. A decision rule chosen on validation data inserts the most certain links
automatically. The repository also contains all compared methods and the scripts behind every table.

## Installation

Python 3.12 was used.

```
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt  # for a GPU, install the PyTorch build for your CUDA version first
```

## Data

| File | Content | Source |
|---|---|---|
| `data/ocel2-p2p.sqlite` | synthetic Procure-to-Pay log, OCEL 2.0 (included) | [ocel-standard.org](https://www.ocel-standard.org/event-logs/overview/) |
| `data/BPI2017-Final.csv` | BPI Challenge 2017, object-centric version (not included, too large) | distributed with the [ocpa](https://github.com/ocpm/ocpa) library; used by Sadeghianasl et al. (2026) |

Convert the BPI 2017 file once:

```
python experiments/build_bpi2017_ocel2.py data/BPI2017-Final.csv data/bpi2017-ocel2.sqlite
python experiments/make_e2o_copy.py data/bpi2017-ocel2.sqlite data/bpi2017-ocel2-e2o.sqlite   # E2O relations only
```

## Quick start

```
python main.py compare data/ocel2-p2p.sqlite --loader sqlite --epochs 2000 --seed 0 \
    --pair-features --time-feature --object-inputs type+degree --results-json results.json
```

This hides one object of 200 sampled test events (partial gaps), trains our model, evaluates it together with the
co-occurrence model and writes Hits@K, MRR, the per-gap ranks and the decision-rule statistics to `results.json`.
`python main.py overview <log>` prints log statistics.

| Option | Effect |
|---|---|
| `--pair-features --time-feature --object-inputs type+degree` | our model (GNN + pairwise evidence) |
| no options | plain GNN (per-object embeddings, cosine score) |
| `--features-only --time-feature --object-inputs type+degree` | pairwise features only (MLP) |
| `--o2o-drop 1.0` | remove all O2O relations |
| `--missing-rate p` | log-wide protocol: remove every E2O link with probability `p` |
| `--context-drop p` | remove a share `p` of the training links |
| `--two-stage-hops 1 2` | additionally rank only candidates within 1 or 2 hops |
| `--one-object-context` | diagnostic of Section 5.2.1: add one-object events to the graphs as context |

## Structure

```
main.py          command-line entry point (run / compare / overview)
core/            OCEL data model, gap protocols, ranking metrics, decision rule, pipeline
adapters/        log loaders (OCEL 2.0 SQLite / JSON), co-occurrence model, GNN (gnn.py, gnn_scalable.py)
experiments/     scripts behind the tables and figures; run_all.bat runs them all
data/            event logs
```

## Reproducing the paper

Run all commands from the repository root; results go to `experiments/results/`.
`experiments/run_all.bat` runs everything except the published detector, one job after another, and skips finished
runs, so it can be restarted (Windows; the commands are the same on other systems). Below, `M` stands for
`--pair-features --time-feature --object-inputs type+degree`.

| Paper | Command(s) |
|---|---|
| Table 1, our model | `python experiments/run_seeds.py data/ocel2-p2p.sqlite --tag p2p_pft --seeds 0 1 2 3 4 -- M --two-stage-hops 1 2`; without O2O: add `--o2o-drop 1.0` and another `--tag` |
| Table 1, plain GNN / features only (MLP) | `run_seeds.py ... -- --two-stage-hops 1 2` / `run_seeds.py ... -- --features-only --time-feature --object-inputs type+degree` |
| Table 1, gradient boosting | `python experiments/gb_features_time.py data/ocel2-p2p.sqlite OUT.json --seeds 0 1 2 3 4 --time [--o2o-drop 1.0]` |
| Table 1, O2O rule, Adamic–Adar, personalized PageRank | `python experiments/graph_heuristics.py data/ocel2-p2p.sqlite OUT.json --seeds 0 1 2 3 4 [--o2o-drop 1.0]` |
| Table 2, BPI 2017 (five splits: default split + seeds 1–4) | default split: `python experiments/run_compare_slim.py compare data/bpi2017-ocel2-e2o.sqlite --loader sqlite --epochs 2000 --results-json OUT.json M --two-stage-hops 1`; seeds: `run_seeds.py data/bpi2017-ocel2-e2o.sqlite --seeds 1 2 3 4 --slim -- ...`; heuristics and gradient boosting with `--default-split` / `--seeds 1 2 3 4` |
| Section 5.2.1, one-object events as context | `run_seeds.py data/bpi2017-ocel2-e2o.sqlite --tag bpi_pft_one_object_context --seeds 1 --slim -- M --two-stage-hops 1 --one-object-context` |
| Table 3, repair rules and co-occurrence | `python experiments/bpi2017_lost_memory_reconstruction.py data/BPI2017-Final.csv OUT.json 0 1 2 3 4` |
| Table 3, our model (with / without φ6) | `python experiments/bpi2017_lost_memory_pft.py data/BPI2017-Final.csv . OUT.json SEED 2000` / `bpi2017_lost_memory_pf.py` (same arguments) |
| Table 3, published detector | see below |
| Section 6.3, erased offers disconnected from their event | `python experiments/components.py data/BPI2017-Final.csv` |
| Section 6.4, sensitivity | log-wide: `run_seeds.py ... -- --missing-rate p M` (p = 0.1, 0.3, 0.5, 0.7); training links removed: `run_seeds.py ... -- M --context-drop p` (p = 0.1, 0.3, 0.5) |
| Section 6.5, object identities and transfer | `run_seeds.py ... --epochs 600 -- --pair-features --time-feature --object-inputs id` (vs. `type+degree`); `python experiments/inductive_transfer.py data/ocel2-p2p.sqlite --seed 0 --epochs 600 M --out OUT.json` |
| Section 6.6 and Figure 1, decision rule and training curves | stored in every run's JSON; insertions on complete events: `python experiments/false_insertions.py data/ocel2-p2p.sqlite experiments/results/seeds_p2p_pft/seed0.json ... --out OUT.json` |
| Section 7, time-proximity rule | `python experiments/time_rule.py data/ocel2-p2p.sqlite OUT.json --seeds 0 1 2 3 4`; BPI 2017: `--default-split` |
| Means, confidence intervals, paired bootstrap | `python experiments/aggregate_seeds.py experiments/results/seeds_p2p_pft`; `python experiments/paired_bootstrap.py experiments/results/seeds_p2p_pft:gnn experiments/results/seeds_p2p_plain:gnn` |

**Published detector.** The detector of Sadeghianasl et al. is re-run with the authors' unchanged code. Clone
[AnomalyDetection_OCIP](https://github.com/jonghyeonk/AnomalyDetection_OCIP) into the repository root, install its
requirements in a separate environment (it needs pandas < 3), and run `python experiments/ocip_lost_memory.py 0 1 2 3 4`.

Model runs take from minutes to a few hours each, depending on the log and the hardware; the results vary slightly
between GPUs.

## License

MIT, see [LICENSE](LICENSE). The P2P log and BPI 2017 are subject to the licenses of their sources.

## Citation

If you use this code, please cite:

> Áron Balázs Papp and István Koren. Object-Centric Event Log Repair Through Graph Neural Networks.
> Submitted to the International Conference on Process Mining (ICPM 2027).
