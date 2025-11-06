import argparse
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
    "gpt-4o":              "GPT-4o",
    "deepseek-chat":       "DeepSeek-V3",
}

JAPANESE_MODELS = {
    "llm-jp-3-13b", "sarashina2-13b",
    "Swallow-13b-hf", "Swallow-MS-7b-v0.1",
    "Llama-3-Swallow-8B-v0.1"
}

PURPLE_MODELS = {"gpt-4o", "deepseek-chat"}
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
        if model_dir_name in PURPLE_MODELS:
            if prompt_lan.startswith("ja"):
                display_name += " (ja)"
            elif prompt_lan.startswith("en"):
                display_name += " (en)"

        row = {}
        overall_acc = summary.get(direction, {}).get("accuracy")
        if overall_acc is not None:
            row[ALL_COL_NAME] = overall_acc

        per_era = summary.get("per_era", {})
        for era in ERA_ORDER:
            acc = per_era.get(era, {}).get(direction, {}).get("accuracy")
            if acc is not None:
                row[era] = acc

        data[display_name] = row

    return data


def draw_heatmap(data_dict: dict, from_calendar: str, to_calendar: str, prompt_lan: str, output_dir: str = "."):
    if not data_dict:
        print(f"[WARN] No data for {prompt_lan} {from_calendar}->{to_calendar}")
        return

    df = pd.DataFrame.from_dict(data_dict, orient="index")

    ordered_rows = []
    for _, val in MODEL_NAME_MAP.items():
        if val in df.index:
            ordered_rows.append(val)
        if f"{val} (ja)" in df.index:
            ordered_rows.append(f"{val} (ja)")
        if f"{val} (en)" in df.index:
            ordered_rows.append(f"{val} (en)")

    df = df.reindex(index=ordered_rows)

    expected_cols = ERA_ORDER + [ALL_COL_NAME]
    df = df.reindex(columns=[c for c in expected_cols if c in df.columns])

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df,
        annot=True,
        cmap="Reds",
        vmin=0, vmax=1,
        fmt=".2f",
        cbar_kws={"label": "Accuracy"},
        annot_kws={"fontsize": 18}
    )
    ax.set_xlabel("Era", labelpad=10, fontsize=20)
    ax.set_ylabel("Model", labelpad=10, fontsize=20)
    ax.set_title(f"{from_calendar} → {to_calendar} Conversion", pad=12, fontsize=20)
    ax.set_xticklabels(
        [c.capitalize() if c != ALL_COL_NAME else "All" for c in df.columns],
        ha="center", fontsize=18, rotation=30
    )
    plt.yticks(rotation=0, fontsize=18)

    for tick_label in ax.get_yticklabels():
        model_name = tick_label.get_text().split(" ")[0]
        dir_name = next((k for k, v in MODEL_NAME_MAP.items() if v == model_name), None)
        if dir_name in JAPANESE_MODELS:
            tick_label.set_color("#d62728")  # Red
        elif dir_name in PURPLE_MODELS:
            tick_label.set_color("#9467bd")  # Purple
        else:
            tick_label.set_color("#1f77b4")  # Blue

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


def main():
    parser = argparse.ArgumentParser(description="Generate accuracy heatmaps for Japanese calendar conversion tasks.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing results")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for heatmaps")
    parser.add_argument("--models", nargs="*", help="Filter model directories (optional)")
    parser.add_argument("--prompts", nargs="+", default=["natural-prompt"], help="Prompt languages")
    args = parser.parse_args()

    for prompt_lan in args.prompts:
        for direction, (from_cal, to_cal) in CONVERSION_LABELS.items():
            if prompt_lan == "natural-prompt":
                data_ja = load_accuracy_data(args.base_dir, "ja-prompt", direction)
                data_en = load_accuracy_data(args.base_dir, "en-prompt", direction)
                renamed = {}
                for k, v in {**data_ja, **data_en}.items():
                    if k in ["GPT-4o", "DeepSeek-V3"]:
                        if k in data_ja:
                            renamed[f"{k} (ja)"] = data_ja[k]
                        if k in data_en:
                            renamed[f"{k} (en)"] = data_en[k]
                    else:
                        renamed[k] = v
                data = renamed
            else:
                data = load_accuracy_data(args.base_dir, prompt_lan, direction, filter_models=args.models)
            draw_heatmap(data, from_cal, to_cal, prompt_lan, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
