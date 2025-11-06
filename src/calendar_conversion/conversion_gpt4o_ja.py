import glob
import json
import os
import re

from openai import OpenAI
from tqdm import tqdm

# 元号境界の許容
BOUNDARY_MAP = {
    ("明治", 45): ("大正", 1),
    ("大正", 1): ("明治", 45),
    ("大正", 15): ("昭和", 1),
    ("昭和", 1): ("大正", 15),
    ("昭和", 64): ("平成", 1),
    ("平成", 1): ("昭和", 64),
    ("平成", 31): ("令和", 1),
    ("令和", 1): ("平成", 31),
}

def is_valid_wareki_format(text):
    return bool(re.search(r"(明治|大正|昭和|平成|令和|安政|文久|慶応|元治|万延|文政|天保|嘉永|文化)[元\d]+年", text))

def is_valid_seireki_format(text):
    return bool(re.search(r"\d{4}年", text))

def is_correct(response, answer, alt_answer=None):
    answer_strs = [str(answer)]
    if answer == 1:
        answer_strs.append("元")
    if alt_answer is not None:
        answer_strs.append(str(alt_answer))
        if alt_answer == 1:
            answer_strs.append("元")
    return any(ans in response for ans in answer_strs)

def generate_batch(prompts, model, client, max_new_tokens=32):
    """ChatGPT API を使ってプロンプトごとに1レスポンスを返す"""
    results = []
    for prompt in tqdm(prompts):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは和暦とグレゴリオ暦の専門家です。以下に続くように文章を答えのみ生成してください。"},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_new_tokens,
            temperature=0.0,
        )
        res = completion.choices[0].message.content.strip()
        results.append(res.split("\n")[0])
    return results

def run(args):
    client = OpenAI(api_key=args.api_key)
    model = "gpt-4o"
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir = output_dir +f"/ja-prompt/{model}"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "wareki": {"correct": 0, "total": 0, "format_match": 0},
        "seireki": {"correct": 0, "total": 0, "format_match": 0},
        "per_era": {}
    }

    for path in sorted(glob.glob(f"{input_dir}/ja/*.jsonl")):
        era = os.path.splitext(os.path.basename(path))[0]
        if era not in {"heisei", "meiji", "taisho", "showa"}:
            continue

        with open(path) as f:
            lines = [json.loads(line) for line in f]

        print(f"{era}: {len(lines)} data")

        summary["per_era"][era] = {
            "wareki": {"correct": 0, "total": 0, "format_match": 0},
            "seireki": {"correct": 0, "total": 0, "format_match": 0},
        }

        prompts_wareki, prompts_seireki, meta_entries = [], [], []

        for entry in lines:
            period = entry["Era"]
            seireki_str = entry["Gregorian"]
            wareki_str = entry["Japanese"]

            m_wareki = re.match(r"(明治|大正|昭和|平成|令和)(元|\d+)年", wareki_str)
            if not m_wareki:
                continue
            period = m_wareki.group(1)
            wareki = 1 if m_wareki.group(2) == "元" else int(m_wareki.group(2))

            m_seireki = re.match(r"(\d{4})年", seireki_str)
            if not m_seireki:
                continue
            seireki = int(m_seireki.group(1))

            alt_period, alt_wareki = BOUNDARY_MAP.get((period, wareki), (None, None))
            alt_seireki = seireki if alt_period else None

            one_shot_wareki = f"1804年を和暦に変換すると、文化元年です。{seireki}年を和暦に変換すると、"
            one_shot_seireki = f"文化1年を西暦に変換すると、1804年です。{period}{wareki}年を西暦に変換すると、"

            prompts_wareki.append(one_shot_wareki)
            prompts_seireki.append(one_shot_seireki)

            meta_entries.append({
                **entry,
                "seireki_value": seireki,
                "wareki_value": wareki,
                "period_label": period,
                "prompt_wareki": one_shot_wareki,
                "prompt_seireki": one_shot_seireki,
                "alt_wareki": alt_wareki,
                "alt_seireki": alt_seireki,
            })

        responses_wareki = generate_batch(prompts_wareki, model, client)
        responses_seireki = generate_batch(prompts_seireki, model, client)

        output_path = os.path.join(output_dir, f"{era}.jsonl")
        with open(output_path, "w") as fout:
            for i, meta in enumerate(meta_entries):
                seireki, wareki = meta["seireki_value"], meta["wareki_value"]
                alt_wareki, alt_seireki = meta["alt_wareki"], meta["alt_seireki"]

                res_wareki, res_seireki = responses_wareki[i], responses_seireki[i]

                is_format_wareki = is_valid_wareki_format(res_wareki)
                is_correct_wareki = is_correct(res_wareki, wareki, alt_wareki) if is_format_wareki else False

                is_format_seireki = is_valid_seireki_format(res_seireki)
                is_correct_seireki = is_correct(res_seireki, seireki, alt_seireki) if is_format_seireki else False

                summary["wareki"]["total"] += 1
                summary["seireki"]["total"] += 1
                if is_format_wareki:
                    summary["wareki"]["format_match"] += 1
                    if is_correct_wareki:
                        summary["wareki"]["correct"] += 1
                if is_format_seireki:
                    summary["seireki"]["format_match"] += 1
                    if is_correct_seireki:
                        summary["seireki"]["correct"] += 1

                summary["per_era"][era]["wareki"]["total"] += 1
                summary["per_era"][era]["seireki"]["total"] += 1
                if is_format_wareki:
                    summary["per_era"][era]["wareki"]["format_match"] += 1
                    if is_correct_wareki:
                        summary["per_era"][era]["wareki"]["correct"] += 1
                if is_format_seireki:
                    summary["per_era"][era]["seireki"]["format_match"] += 1
                    if is_correct_seireki:
                        summary["per_era"][era]["seireki"]["correct"] += 1

                fout.write(json.dumps({
                    **meta,
                    "response_wareki": res_wareki,
                    "format_wareki": is_format_wareki,
                    "correct_wareki": is_correct_wareki,
                    "response_seireki": res_seireki,
                    "format_seireki": is_format_seireki,
                    "correct_seireki": is_correct_seireki,
                }, ensure_ascii=False) + "\n")

    for mode in ["wareki", "seireki"]:
        fmt = summary[mode]
        fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
        fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4) if fmt["total"] > 0 else 0.0

    for era in summary["per_era"]:
        for mode in ["wareki", "seireki"]:
            fmt = summary["per_era"][era][mode]
            fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
            fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4) if fmt["total"] > 0 else 0.0

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
