import gc
import glob
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_names = [
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Mistral-7B-v0.1",
    "Llama-3.1-8B",
]

ERA_TRANSLATION = {
    "明治": "Meiji", "大正": "Taisho", "昭和": "Showa", "平成": "Heisei", "令和": "Reiwa",
}
ERA_LATIN_MACRON = {
    "明治": "Meiji", "大正": "Taishō", "昭和": "Shōwa", "平成": "Heisei", "令和": "Reiwa",
}

BOUNDARY_MAP = {
    ("明治", 45): ("大正", 1), ("大正", 1): ("明治", 45),
    ("大正", 15): ("昭和", 1), ("昭和", 1): ("大正", 15),
    ("昭和", 64): ("平成", 1), ("平成", 1): ("昭和", 64),
    ("平成", 31): ("令和", 1), ("令和", 1): ("平成", 31),
}

def is_valid_wareki_format(text):
    return bool(re.search(
        r"(Meiji|Taisho|Taishō|Showa|Shōwa|Heisei|Reiwa|"
        r"Ansei|Bunkyū|Bunkyu|Keiō|Keio|Genji|Man'en|Manen|"
        r"Bunsei|Tenpō|Tenpo|Kaei|Bunka)"
        r"\s(Gannen|\d{1,2})", text, re.IGNORECASE
    ))

def is_valid_seireki_format(text):
    return bool(re.search(r"\b\d{4}\b", text))

def load_model_and_tokenizer(model_name):
    model_path = f"/work00/share/hf_models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_batch(prompts, model, tokenizer, batch_size, max_new_tokens=32):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            outputs = model.generate(
                **generate_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
            decoded = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
            results.append(decoded.split("\n")[0].strip())
    return results

def run_for_model(model_name, input_dir, output_dir, batch_size):
    def is_correct_wareki(response, answer, alt_answer=None):
        answer_strs = []
        for era in [
            "Meiji", "Taisho", "Taishō", "Showa", "Shōwa", "Heisei", "Reiwa",
            "Ansei", "Bunkyū", "Bunkyu", "Keiō", "Keio", "Genji",
            "Man'en", "Manen", "Bunsei", "Tenpō", "Tenpo", "Kaei", "Bunka"
        ]:
            if answer == 1:
                answer_strs.append(f"{era} Gannen")
            answer_strs.append(f"{era} {answer}")
            if alt_answer is not None:
                if alt_answer == 1:
                    answer_strs.append(f"{era} Gannen")
                answer_strs.append(f"{era} {alt_answer}")
        return any(ans in response for ans in answer_strs)

    def is_correct_seireki(response, answer, alt_answer=None):
        patterns = [fr"\b{re.escape(str(answer))}\b"]
        if alt_answer is not None:
            patterns.append(fr"\b{re.escape(str(alt_answer))}\b")
        return any(re.search(p, response) for p in patterns)

    print(f"\n=== Running model: {model_name} ===")
    model, tokenizer = load_model_and_tokenizer(model_name)
    output_dir = f"{output_dir}/en-prompt/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "wareki": {"correct": 0, "total": 0, "format_match": 0},
        "seireki": {"correct": 0, "total": 0, "format_match": 0},
        "per_era": {}
    }

    for path in sorted(glob.glob(f"{input_dir}/*.jsonl")):
        era_file = os.path.splitext(os.path.basename(path))[0]
        if era_file not in {"meiji", "taisho", "showa", "heisei"}:
            continue

        with open(path) as f:
            lines = [json.loads(line) for line in f]

        print(f" {era_file}: {len(lines)}件読み込み")
        summary["per_era"][era_file] = {
            "wareki": {"correct": 0, "total": 0, "format_match": 0},
            "seireki": {"correct": 0, "total": 0, "format_match": 0},
        }

        prompts_wareki, prompts_seireki, meta_entries = [], [], []
        for entry in lines:
            lang = entry.get("Lang", "en")
            if lang != "en":
                continue  # 英語以外はスキップ

            era_eng = entry["Era"]              # 例: "Heisei"
            seireki = entry["Gregorian"]        # 例: 1989
            wareki_str = entry["Japanese"]      # 例: "Heisei 1"

            m = re.match(r"([A-Za-z]+)\s(Gannen|\d+)", wareki_str)
            if not m:
                continue
            era_eng_parsed = m.group(1)
            wareki = 1 if m.group(2).lower() == "gannen" else int(m.group(2))

            # 和暦→日本語元号に逆引き
            jp_period = None
            for jp, en in ERA_TRANSLATION.items():
                if en == era_eng_parsed:
                    jp_period = jp
                    break
            if jp_period is None:
                continue

            roman_period = ERA_LATIN_MACRON[jp_period]
            alt_period, alt_wareki = BOUNDARY_MAP.get((jp_period, wareki), (None, None))
            alt_seireki = seireki if alt_period else None

            prompt_wareki = f"In the Japanese calendar, The year 1804 corresponds to Bunka 1. In the Japanese calendar, The year {seireki} corresponds to"
            prompt_seireki = f"In the Gregorian calendar, Bunka 1 corresponds to the year 1804. In the Gregorian calendar, {roman_period} {wareki} corresponds to the year"

            prompts_wareki.append(prompt_wareki)
            prompts_seireki.append(prompt_seireki)
            meta_entries.append({
                **entry,
                "seireki_value": seireki,
                "wareki_value": wareki,
                "period_label": jp_period,
                "prompt_wareki": prompt_wareki,
                "prompt_seireki": prompt_seireki,
                "alt_wareki": alt_wareki,
                "alt_seireki": alt_seireki,
                "eng_period": era_eng,
                "roman_period": roman_period
            })

        responses_wareki = generate_batch(prompts_wareki, model, tokenizer, batch_size)
        responses_seireki = generate_batch(prompts_seireki, model, tokenizer, batch_size)

        output_path = os.path.join(output_dir, f"{era_file}.jsonl")
        with open(output_path, "w") as fout:
            for i, meta in enumerate(meta_entries):
                seireki = meta["seireki_value"]
                wareki = meta["wareki_value"]
                alt_wareki = meta["alt_wareki"]
                alt_seireki = meta["alt_seireki"]

                res_wareki = responses_wareki[i]
                res_seireki = responses_seireki[i]

                is_format_wareki = is_valid_wareki_format(res_wareki)
                is_correct_wareki_flag = is_correct_wareki(res_wareki, wareki, alt_wareki) if is_format_wareki else False

                is_format_seireki = is_valid_seireki_format(res_seireki)
                is_correct_seireki_flag = is_correct_seireki(res_seireki, seireki, alt_seireki) if is_format_seireki else False

                for mode, is_format, is_correct_flag in [
                    ("wareki", is_format_wareki, is_correct_wareki_flag),
                    ("seireki", is_format_seireki, is_correct_seireki_flag)
                ]:
                    summary[mode]["total"] += 1
                    summary["per_era"][era_file][mode]["total"] += 1
                    if is_format:
                        summary[mode]["format_match"] += 1
                        summary["per_era"][era_file][mode]["format_match"] += 1
                        if is_correct_flag:
                            summary[mode]["correct"] += 1
                            summary["per_era"][era_file][mode]["correct"] += 1

                fout.write(json.dumps({
                    **meta,
                    "response_wareki": res_wareki,
                    "format_wareki": is_format_wareki,
                    "correct_wareki": is_correct_wareki_flag,
                    "response_seireki": res_seireki,
                    "format_seireki": is_format_seireki,
                    "correct_seireki": is_correct_seireki_flag,
                }, ensure_ascii=False) + "\n")

    for mode in ["wareki", "seireki"]:
        fmt = summary[mode]
        fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
        fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4)

    for era in summary["per_era"]:
        for mode in ["wareki", "seireki"]:
            fmt = summary["per_era"][era][mode]
            fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
            fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def run(args):
    for model_name in model_names:
        run_for_model(model_name, args.input_dir, args.output_dir, args.batch_size)
