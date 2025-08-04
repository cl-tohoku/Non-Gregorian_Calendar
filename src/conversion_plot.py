import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_NAME_MAP = {
    "llm-jp-3-13b":        "LLM-JP-3-13B",
    "sarashina2-13b":      "Sarashina2-13B",
    "Swallow-13b-hf":      "Swallow-13B",
    "Swallow-MS-7b-v0.1":  "Swallow-MS-7B",
    "Llama-3-Swallow-8B-v0.1": "LLaMA3-Swallow-8B",
    "Llama-2-7b-hf":       "LLaMA2-7B",
    "Llama-2-13b-hf":      "LLaMA2-13B",
    "Mistral-7B-v0.1":     "Mistral-7B",
    "Llama-3.1-8B":        "LLaMA3.1-8B",
}

JAPANESE_MODELS = {
    "llm-jp-3-13b",
    "sarashina2-13b",
    "Swallow-13b-hf",
    "Swallow-MS-7b-v0.1",
    "Llama-3-Swallow-8B-v0.1"
}

ERA_ORDER = ["meiji", "taisho", "showa", "heisei"]
ALL_COL_NAME = "all"

CONVERSION_LABELS = {
    "wareki":  ("Gregorian", "Japanese"),
    "seireki": ("Japanese", "Gregorian"),
}


def load_accuracy_data(base_dir: str, prompt_lan: str, direction: str, filter_models=None) -> dict:
    data = {}
    dir_path = os.path.join(base_dir, prompt_lan)
    if not os.path.isdir(dir_path):
        return data

    for model_dir_name in os.listdir(dir_path):
        if filter_models is not None and model_dir_name not in filter_models:
            continue
        if model_dir_name not in MODEL_NAME_MAP:
            continue
        summary_path = os.path.join(dir_path, model_dir_name, "summary.json")
        if not os.path.isfile(summary_path):
            continue

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        display_name = MODEL_NAME_MAP[model_dir_name]
        row = {}

        overall_acc = summary.get(direction, {}).get("accuracy")
        if overall_acc is not None:
            row[ALL_COL_NAME] = overall_acc

        per_era = summary.get("per_era", {})
        for era in ERA_ORDER:
            acc = (
                per_era
                .get(era, {})
                .get(direction, {})
                .get("accuracy")
            )
            if acc is not None:
                row[era] = acc

        data[display_name] = row

    return data

def draw_heatmap(data_dict: dict,
                 from_calendar: str,
                 to_calendar: str,
                 prompt_lan: str,
                 output_dir: str = "."):
    if not data_dict:
        print(f"[WARN] No data for {prompt_lan} {from_calendar}->{to_calendar}")
        return

    df = pd.DataFrame.from_dict(data_dict, orient="index")

    ordered_rows = [MODEL_NAME_MAP[k] for k in MODEL_NAME_MAP if MODEL_NAME_MAP[k] in df.index]
    df = df.reindex(index=ordered_rows)

    expected_cols = ERA_ORDER + [ALL_COL_NAME]
    existing_cols = [c for c in expected_cols if c in df.columns]
    df = df.reindex(columns=existing_cols)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df,
        annot=True,
        cmap="Reds",
        vmin=0,
        vmax=1,
        fmt=".2f",
        cbar_kws={"label": "Accuracy"},
        annot_kws={"fontsize": 18}
    )
    ax.set_xlabel("Era", labelpad=10, fontsize=20)
    ax.set_ylabel("Model", labelpad=10, fontsize=20)

    title = f"{from_calendar} → {to_calendar} Conversion"
    ax.set_title(title, pad=12, fontsize=20)

    ax.set_xticklabels([c.capitalize() if c != ALL_COL_NAME else "All" for c in existing_cols], ha="center", fontsize=18, rotation=30)
    plt.yticks(rotation=0, fontsize=18)

    # === Y軸ラベル（モデル名）のフォントサイズ + 色分け ===
    for tick_label in ax.get_yticklabels():
        model_disp_name = tick_label.get_text()
        # model_dir_name を逆引き（モデル表示名からディレクトリ名へ）
        model_dir_name = next((k for k, v in MODEL_NAME_MAP.items() if v == model_disp_name), None)
        if model_dir_name in JAPANESE_MODELS:
            tick_label.set_color("#d62728")  # 赤
        else:
            tick_label.set_color("#1f77b4")  # 青

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(20)
    cbar.ax.tick_params(labelsize=20)

    os.makedirs(output_dir, exist_ok=True)
    filename_base = f"{prompt_lan}_{from_calendar}_to_{to_calendar}_accuracy_heatmap"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_base}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{filename_base}.pdf"))
    plt.close()
    print(f"[INFO] saved: {filename_base}.png/.pdf")

def run(args):
    base_dir = args.output_dir

    for direction, (from_cal, to_cal) in CONVERSION_LABELS.items():
        data_ja = load_accuracy_data(base_dir, "ja-prompt", direction, filter_models=JAPANESE_MODELS)
        data_en = load_accuracy_data(base_dir, "en-prompt", direction,
                                        filter_models=set(MODEL_NAME_MAP.keys()) - JAPANESE_MODELS)
        data = {**data_ja, **data_en}
        draw_heatmap(data, from_cal, to_cal, output_dir=base_dir)
