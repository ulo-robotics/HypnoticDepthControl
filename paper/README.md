# Paper analysis

The tables and figures of the paper and its supplement, from the stored outputs of the run (commands in the supplement's
reproducibility section). Run from the repository root.

On the GPU machine, after the evaluations (`runs/finetune/eval30`, `eval30_mpc`, `eval250`, `eval250_mpc`):

    python paper/traces.py runs/finetune [old_long.json]   # trace quantiles, cost per 30-min segment over 250 min
    python paper/model_info.py runs/finetune/best_model.pkl
    python paper/covariates.py --run runs/finetune         # UKF-MM / MPC-MM with the patients' individual PK (CPU, hours)

Then collect one data folder (copied, nothing recomputed):

    DATA/search/       runs/search                      DATA/search.log      the search's stdout
    DATA/finetune/     runs/finetune (log.jsonl, ...)   DATA/finetune.log    the fine-tuning's stdout
    DATA/eval30/       runs/finetune/eval30             DATA/mpc30/          runs/finetune/eval30_mpc
    DATA/eval250/      runs/finetune/eval250            DATA/mpc250/         runs/finetune/eval250_mpc
    DATA/covariates/   runs/finetune/paper/covariates   DATA/covariates.json
    DATA/long_segments.json, DATA/model_info.json       runs/finetune/paper
    DATA/onnx_export.log, DATA/onnx_export_litert.log   export_onnx.py's stdout
    DATA/edge/         edge.py's results (<host>.jsonl, coral_rpi4.jsonl, nano_trt.jsonl, lethe_*.txt)

and generate (numpy and matplotlib only):

    python paper/analyze.py DATA SUPPLEMENT_DIR   # supplement: tab_*.tex, fig/*.pdf, numbers.json
    python paper/ecc.py DATA PAPER_DIR            # paper: tab_main, tab_ladder, tab_long, tab_edge, fig/long_col.pdf

`old_long.json` holds the per-segment costs of the earlier model with the elapsed-time input (the dashed curves of the long-horizon
figure); it is an output of that model, which this code no longer loads. Without it the figure has no dashed curve.
