@echo off
REM Reproduces every result of the paper, one job after another (Windows; same commands elsewhere).
REM Start from anywhere: experiments\run_all.bat. Finished seeds/files are skipped, so it can be restarted.
REM Requires data\ocel2-p2p.sqlite, data\bpi2017-ocel2-e2o.sqlite and data\BPI2017-Final.csv (see README).
cd /d "%~dp0\.."
set R=experiments\results
if not exist %R% mkdir %R%
set M=--pair-features --time-feature --object-inputs type+degree

REM ---- P2P (Table 1, Section 6.4 sensitivity, Section 7 time rule) ----
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_pft --seeds 0 1 2 3 4 --epochs 2000 -- %M% --two-stage-hops 1 2
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_pft_o2o1.0 --seeds 0 1 2 3 4 --epochs 2000 -- %M% --o2o-drop 1.0 --two-stage-hops 1 2
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_plain --seeds 0 1 2 3 4 --epochs 2000 -- --two-stage-hops 1 2
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_plain_o2o1.0 --seeds 0 1 2 3 4 --epochs 2000 -- --o2o-drop 1.0
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_fot --seeds 0 1 2 3 4 --epochs 2000 -- --features-only --time-feature --object-inputs type+degree
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_fot_o2o1.0 --seeds 0 1 2 3 4 --epochs 2000 -- --features-only --time-feature --object-inputs type+degree --o2o-drop 1.0
for %%p in (0.1 0.3 0.5) do python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_pft_ctx%%p --seeds 0 1 2 3 4 --epochs 2000 -- %M% --context-drop %%p
for %%p in (0.1 0.3 0.5 0.7) do python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_logwide_p%%p --seeds 0 1 2 3 4 --epochs 2000 -- --missing-rate %%p %M%
for %%p in (0.1 0.3 0.5) do python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_plain_ctx%%p --seeds 0 1 2 3 4 --epochs 2000 -- --context-drop %%p
if not exist %R%\p2p_heur.json python experiments\graph_heuristics.py data\ocel2-p2p.sqlite %R%\p2p_heur.json --seeds 0 1 2 3 4
if not exist %R%\p2p_o2o1.0_heur.json python experiments\graph_heuristics.py data\ocel2-p2p.sqlite %R%\p2p_o2o1.0_heur.json --seeds 0 1 2 3 4 --o2o-drop 1.0
if not exist %R%\gb_time_p2p.json python experiments\gb_features_time.py data\ocel2-p2p.sqlite %R%\gb_time_p2p.json --seeds 0 1 2 3 4 --time
if not exist %R%\time_rule_p2p.json python experiments\time_rule.py data\ocel2-p2p.sqlite %R%\time_rule_p2p.json --seeds 0 1 2 3 4
if not exist %R%\gb_time_p2p_o2o1.0.json python experiments\gb_features_time.py data\ocel2-p2p.sqlite %R%\gb_time_p2p_o2o1.0.json --seeds 0 1 2 3 4 --time --o2o-drop 1.0

REM ---- Section 6.5: object identities and transfer ----
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_pft_id600 --seeds 0 1 2 3 4 --epochs 600 -- --pair-features --time-feature --object-inputs id
python experiments\run_seeds.py data\ocel2-p2p.sqlite --tag p2p_pft_td600 --seeds 0 1 2 3 4 --epochs 600 -- %M%
for %%s in (0 1 2 3 4) do if not exist %R%\transfer_pft_seed%%s.json python experiments\inductive_transfer.py data\ocel2-p2p.sqlite --seed %%s --epochs 600 %M% --out %R%\transfer_pft_seed%%s.json

REM ---- BPI 2017 (Table 2; five splits = default split + seeds 1-4) ----
if not exist %R%\bpi2017_partial_gaps_pft.json python experiments\run_compare_slim.py compare data\bpi2017-ocel2-e2o.sqlite --loader sqlite --epochs 2000 --results-json %R%\bpi2017_partial_gaps_pft.json %M% --two-stage-hops 1 > %R%\bpi2017_partial_gaps_pft.log 2>&1
python experiments\run_seeds.py data\bpi2017-ocel2-e2o.sqlite --tag bpi_pft --seeds 1 2 3 4 --epochs 2000 --slim -- %M% --two-stage-hops 1
if not exist %R%\bpi2017_partial_gaps.json python experiments\run_compare_slim.py compare data\bpi2017-ocel2-e2o.sqlite --loader sqlite --epochs 2000 --results-json %R%\bpi2017_partial_gaps.json > %R%\bpi2017_partial_gaps.log 2>&1
python experiments\run_seeds.py data\bpi2017-ocel2-e2o.sqlite --tag bpi_plain --seeds 1 2 3 4 --epochs 2000 --slim --
if not exist %R%\time_rule_bpi.json python experiments\time_rule.py data\bpi2017-ocel2-e2o.sqlite %R%\time_rule_bpi.json --default-split
REM diagnostic of Section 5.2.1: one-object events added to the graphs as context (compare with seeds_bpi_pft\seed1)
python experiments\run_seeds.py data\bpi2017-ocel2-e2o.sqlite --tag bpi_pft_one_object_context --seeds 1 --epochs 2000 --slim -- %M% --two-stage-hops 1 --one-object-context
if not exist %R%\gb_time_bpi_s1234.json python experiments\gb_features_time.py data\bpi2017-ocel2-e2o.sqlite %R%\gb_time_bpi_s1234.json --seeds 1 2 3 4 --time
if not exist %R%\bpi_heur_default.json python experiments\graph_heuristics.py data\bpi2017-ocel2-e2o.sqlite %R%\bpi_heur_default.json --default-split
if not exist %R%\gb_time_bpi.json python experiments\gb_features_time.py data\bpi2017-ocel2-e2o.sqlite %R%\gb_time_bpi.json --default-split --time

REM ---- Lost Memory benchmark (Table 3, Section 6.3) ----
if not exist %R%\bpi2017_lost_memory_rules.json python experiments\bpi2017_lost_memory_reconstruction.py data\BPI2017-Final.csv %R%\bpi2017_lost_memory_rules.json 0 1 2 3 4
for %%s in (0 1 2 3 4) do if not exist %R%\bpi2017_lost_memory_pft_seed%%s.json python experiments\bpi2017_lost_memory_pft.py data\BPI2017-Final.csv . %R%\bpi2017_lost_memory_pft_seed%%s.json %%s 2000 > %R%\bpi2017_lost_memory_pft_seed%%s.log 2>&1
if not exist %R%\bpi2017_lost_memory_pf_seed0.json python experiments\bpi2017_lost_memory_pf.py data\BPI2017-Final.csv . %R%\bpi2017_lost_memory_pf_seed0.json 0 2000 > %R%\bpi2017_lost_memory_pf_seed0.log 2>&1

python experiments\components.py data\BPI2017-Final.csv > %R%\lost_memory_components.txt

REM ---- Section 6.6: decision rule on complete events ----
if not exist %R%\false_insertions_p2p.json python experiments\false_insertions.py data\ocel2-p2p.sqlite %R%\seeds_p2p_pft\seed0.json %R%\seeds_p2p_pft\seed1.json %R%\seeds_p2p_pft\seed2.json %R%\seeds_p2p_pft\seed3.json %R%\seeds_p2p_pft\seed4.json --out %R%\false_insertions_p2p.json
if not exist %R%\false_insertions_bpi.json python experiments\false_insertions.py data\bpi2017-ocel2-e2o.sqlite %R%\bpi2017_partial_gaps_pft.json %R%\seeds_bpi_pft\seed1.json %R%\seeds_bpi_pft\seed2.json --slim --out %R%\false_insertions_bpi.json

echo ALL DONE
