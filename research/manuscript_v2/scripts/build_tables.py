#!/usr/bin/env python3
"""Format archived CSVs into V2 tables; no model/data access or new diagnostics."""
import csv
import hashlib
import io
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/manuscript_v2/tables"
TEX = ROOT / "论文投稿/manuscript_v2/tables"
SOURCES = {}


def load(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(io.StringIO(path.read_text())))


def export(name, rows, note):
    columns = list(rows[0])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / (name + ".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    text = "# " + name + "\n\n" + note + "\n\n"
    text += "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in rows:
        text += "| " + " | ".join(str(row[x]).replace("|", "\\|") for x in columns) + " |\n"
    (OUT / (name + ".md")).write_text(text)


def escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def latex(name, headers, rows, caption, note):
    TEX.mkdir(parents=True, exist_ok=True)
    text = "% Generated from frozen sources by build_tables.py; not a prose draft.\n"
    text += "\\begin{table}[htbp]\n\\centering\\small\n"
    text += "\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}\n\\toprule\n"
    text += " & ".join(headers) + " \\\\\n\\midrule\n"
    for row in rows:
        text += " & ".join(row) + " \\\\\n"
    text += "\\bottomrule\n\\end{tabular}\n"
    text += "\\caption{" + caption + "}\n\\label{tab:" + name + "}\n"
    text += "\\par\\smallskip\\begin{minipage}{0.96\\linewidth}\\footnotesize " + note + "\\end{minipage}\n"
    text += "\\end{table}\n"
    (TEX / (name + ".tex")).write_text(text)


def main():
    archived = {}
    for short in ["cross_domain_transfer", "teacher_quality_intervention", "teacher_source_ablation", "tt_vs_tg", "efficiency", "failed_or_negative_controls"]:
        rows = load("research/final_evidence/table_" + short + ".csv")
        archived[short] = rows
        note = "Frozen Phase-2 snapshot, copied without changing values. Source: research/final_evidence/table_" + short + ".csv."
        if short in ("teacher_quality_intervention", "teacher_source_ablation"):
            note += " Teacher NLL scope: the archived 512-validation-chunk subset (130560 predicted targets), not the full-split Phase-2F evaluation."
        if short == "tt_vs_tg":
            note += " Teacher NLL/JSD/agreement use the common full validation split. TT/TG are not strictly parameter/training matched; alpha learning rate differs."
        if short in ("efficiency", "failed_or_negative_controls"):
            note += " Legacy/negative controls only: do not pool into confirmatory tables."
        export("table_" + short, rows, note)
    rows = archived["cross_domain_transfer"]
    latex("cross_domain_transfer", ["Dataset", "CE NLL", "KD NLL", "Gain (mean $\\pm$ SD)", "Signs"],
          [[escape(x["dataset"]), x["ws_ce_nll_mean"], x["ws_kd_nll_mean"],
            x["kd_gain_nll_mean"] + " $\\pm$ " + x["kd_gain_nll_sample_sd"], "3/3 " + ("+" if "positive" in x["sign_consistency"] else "--")] for x in rows],
          "Paired warm-start transfer across three text domains (planning table).",
          "Three independently trained student initializations per dataset; one frozen teacher per dataset. Gain is CE minus KD test NLL. Sample SD is over paired gains, not an SE. $\\lambda=5$, $T=2$, token-mean KD; validation-selected endpoints. Cross-domain comparisons do not isolate a causal dataset effect.")
    rows = archived["teacher_quality_intervention"]
    latex("teacher_state", ["Teacher/state contrast", "Teacher val NLL", "Gain/contrast", "Sample SD", "Signs"],
          [["Joint J", rows[0]["teacher_validation_nll"], rows[0]["mean_gain_or_contrast"], rows[0]["sample_sd"], "3/3 +"],
           ["Alpha-only A", rows[1]["teacher_validation_nll"], rows[1]["mean_gain_or_contrast"], rows[1]["sample_sd"], "3/3 --"],
           ["$Q$ (student test contrast)", "--", rows[2]["mean_gain_or_contrast"], rows[2]["sample_sd"], "3/3 +"]],
          "Fixed-composition WT2 teacher-state intervention (planning table).",
          "$Q$ is the KD-student test NLL under A minus that under J, not the teacher NLL difference. Teacher NLL uses the archived 512-chunk validation subset. Forced fusion weight is 0.5. Changing training state also changes branch representations; this is not a scalar-NLL causal intervention. The failed short smoke and explicitly amended full-data gate must remain in the protocol appendix.")
    rows = archived["teacher_source_ablation"]
    latex("teacher_sources", ["Source", "$\\alpha$", "Teacher val NLL", "Gain (mean $\\pm$ SD)"],
          [[escape(x["label"]), x["alpha"], x["teacher_validation_nll"], x["kd_gain_mean"] + " $\\pm$ " + x["kd_gain_sample_sd"]] for x in rows if x["row_type"] == "condition"],
          "WT2 fixed-source ablation (planning table).",
          "Three paired students; teacher NLL is on the archived validation subset. Fused minus Transformer gain: $0.020797\\pm0.001214$; fused minus Grassmann: $0.114674\\pm0.001001$, both 3/3 positive. Transformer-only retains \\texttt{CLIP\\_SATURATION\\_WARNING}; quality and composition also vary. No geometry-specific benefit is supported.")
    rows = archived["tt_vs_tg"][:2]
    latex("tt_vs_tg", ["Teacher", "Parameters", "Val NLL", "JSD", "Agreement", "KD gain"],
          [[x["teacher_or_contrast"], f'{int(x["parameters"]):,}', x["validation_nll"], x["branch_jsd"], x["top1_agreement"], x["kd_gain_mean"]] for x in rows],
          "Homogeneous TT control versus heterogeneous TG (planning table).",
          "Common full validation split for teacher metrics. $H=\\mathrm{NLL}_{TT}-\\mathrm{NLL}_{TG}=-0.012460\\pm0.001348$, TT better for 3/3 students. TT is about 6.38\\% smaller, has 0.010108 lower teacher NLL, and uses a different alpha learning rate (0.01 versus historical TG 0.005). This rejects a TG superiority claim here, not all heterogeneous ensembles.")
    residual = [dict(seed=x["Seed"], teacher_validation_subset_nll=x["Teacher validation NLL"],
                     comparator="validation-selected_C0_CE_on_same_subset", c0_validation_subset_nll=x["C0 validation-subset NLL"],
                     teacher_residual_vs_c0=x["Delta_teacher"], test_gain=x["Delta_KD"])
                for x in load("research/experiments/phase2c_multiseed_teacher_utility/results_multiseed.csv") if x["Arm"] == "C3"]
    export("table_weaker_teacher", residual, "Globally weaker means worse held-out likelihood than the validation-selected matched C0 CE endpoint on the same subset, not automatically worse than S0. No weak-teacher superiority claim.")
    latex("weaker_teacher", ["Seed", "Teacher residual vs C0", "KD test gain"],
          [[x["seed"], f'{float(x["teacher_residual_vs_c0"]):.6f}', f'{float(x["test_gain"]):.6f}'] for x in residual],
          "A globally weaker teacher can still transfer positively (planning table).",
          "Residual is C0 validation-subset NLL minus teacher NLL. All residuals are negative while gains are positive. This refutes a universal necessity rule; it does not show that weak teachers are usually better or that likelihood is irrelevant.")
    per_seed = load("research/experiments/phase3a_safe_lambda_diagnostic/condition_summary.csv")
    export("table_endpoint_inventory", [dict(condition_id=x["condition_id"], dataset=x["dataset"], family=x["family"],
                                           student_seed=x["student_seed"], test_gain=x["Gain_KD"]) for x in per_seed],
           "Archived endpoint inventory: 21 seed-level conditions, seven families, with shared students/controls across WT2 families. Not 21 independent replications. Sources remain Phase-2 endpoints, not new Phase-3 endpoint measurements.")
    predictors = load("research/experiments/phase3a_safe_lambda_diagnostic/predictor_metrics.csv")
    export("table_local_diagnostics", predictors, "Exploratory post-freeze Phase-3A offline diagnostic. Seven families share student states; calibration subsets are not extra seeds. Correlations/AUROCs across corpora cannot establish a universal mechanism.")
    selected = [x for x in predictors if x["level"] == "seed" and x["predictor"] in ("Validation dot a", "Quadratic gain", "Virtual nominal gain")]
    latex("local_diagnostics", ["Offline predictor", "Conditions", "AUROC", "Sign accuracy", "Spearman"],
          [[escape(x["predictor"]), x["n"], f'{float(x["auroc"]):.6f}', f'{float(x["sign_accuracy"]):.6f}', f'{float(x["spearman_rho"]):.6f}'] for x in selected],
          "Local diagnostics do not reliably predict endpoint transfer (planning table).",
          "21 conditions in seven dependent families; three calibration subsets each, not 63 independent runs. First/second-order signs agree on 63/63 calibration rows, but are calibration-stable for only 11/21 conditions. Curvature adds no sign separation. Negative diagnostic, not a proposed successful Safe-lambda method.")
    phase4b = load("research/experiments/phase4b_fineweb_multiseed_replication/results_multiseed.csv")
    export("table_modern_per_seed", phase4b, "Fixed FineWeb-Edu/SmolLM2 stress test. Seed42 reused from Phase4A, not counted twice. Three independent adaptation seeds from the same pretrained base; all arms prefer S0 when step0 is allowed. Not clean modern confirmation.")
    stats = []
    for strength in (1, 5):
        values = [float(x[f"Gain_{strength}"]) for x in phase4b]
        stats.append(dict(dataset="FineWeb-Edu", strength=strength, seeds=3,
                          mean_gain=statistics.mean(values), sample_sd=statistics.stdev(values),
                          sign_consistency="3/3_negative", validation_test_agreement="3/3",
                          best_including_s0="9/9_arms_select_S0", evidence_role="bounded_stress_test",
                          warning="CLIP_SATURATION_WARNING" if strength == 5 else "frequent_clipping"))
    export("table_modern_stress", stats, "Primary trained-checkpoint endpoints remain frozen; secondary step0 selection gives zero diagnostic gains. CE itself worsens all S0 test endpoints. Do not headline this as clean modern KD failure.")
    latex("modern_stress", ["Strength", "Mean test gain", "Sample SD", "Negative signs"],
          [[str(x["strength"]), f'{x["mean_gain"]:.6f}', f'{x["sample_sd"]:.6f}', "3/3"] for x in stats],
          "Bounded modern-model stress test, not clean confirmation (planning table).",
          "All 9/9 step-0-eligible choices select S0; CE also worsens S0 test NLL (mean change $+0.007918$). All lambda-5 arms retain \\texttt{CLIP\\_SATURATION\\_WARNING}; lambda-1 clipping is 0.798526--0.854218. Teacher advantage, pretraining overlap and finite CE456 gradient spike remain confounds. Phase4C did not resolve continuation degeneracy.")
    gate = load("research/experiments/phase4c_nondegenerate_modern_control/ce_lr_gate.csv")
    export("table_modern_ce_gate", gate, "One seed42, three CE LR candidates on an 8M-target disjoint stream. Gate improvement >=0.01 failed. No KD or new test; missing quantities are NOT_RUN, not zeros. Small positive validation headroom is acknowledged.")
    latex("modern_ce_gate", ["CE peak LR", "Selected val NLL", "Val improvement", "Selected step", "Gate"],
          [[x["lr"], f'{float(x["best_validation_nll"]):.6f}', f'{float(x["CE_improvement_validation"]):.6f}', x["selected_step"], "FAIL"] for x in gate],
          "Failed preregistered non-degenerate continuation gate (planning table).",
          "One student seed, not three replications. Best validation improvement $0.004849292809<0.01$; trained step 977 is selected for all candidates. \\texttt{STOP\\_MODERN\\_CONTINUATION\\_NOT\\_ESTABLISHED}. No KD or new test endpoint was measured.")
    for name, relative in [
        ("table_modern_best_including_s0", "research/experiments/phase4b_fineweb_multiseed_replication/best_including_s0.csv"),
        ("table_modern_optimization", "research/experiments/phase4b_fineweb_multiseed_replication/optimization_audit.csv"),
        ("table_modern_gate_optimization", "research/experiments/phase4c_nondegenerate_modern_control/optimization_audit.csv"),
        ("table_tt_vs_tg_per_seed", "research/experiments/phase2f_homogeneous_ensemble_control/results_multiseed.csv"),
    ]:
        export(name, load(relative), "Appendix/source audit only. Archived observations; no new training or model evaluation. Read the corresponding report and warning registry before interpretation.")
    expected = {"research/final_evidence/table_cross_domain_transfer.csv", "research/experiments/phase4c_nondegenerate_modern_control/ce_lr_gate.csv"}
    assert expected <= set(SOURCES)
    assert all(float(x["CE_improvement_validation"]) < 0.01 for x in gate)
    assert all(x["test_evaluated"] == "False" for x in gate)
    for x in phase4b:
        assert abs(float(x["ce_test_nll"]) - float(x["kd1_test_nll"]) - float(x["Gain_1"])) < 1e-12
        assert abs(float(x["ce_test_nll"]) - float(x["kd5_test_nll"]) - float(x["Gain_5"])) < 1e-12
    (OUT.parent / "table_source_manifest.json").write_text(json.dumps(dict(
        created_on="2026-10-04", activity="archived CSV formatting/descriptive checks only",
        sources=SOURCES, gain_sign_checks="PASS", independent_seed_unit="student initialization or adaptation, not corpus/teacher/pretraining replication"), indent=2) + "\n")
    print(f"Formatted archived tables; {len(SOURCES)} hashed CSV sources; checks PASS.")


if __name__ == "__main__":
    main()
