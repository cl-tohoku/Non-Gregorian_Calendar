import argparse

import conversion_gpt4o_en
import conversion_gpt4o_ja
import conversion_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key for GPT-4o")
    args = parser.parse_args()

    conversion_gpt4o_en.run(args)
    conversion_gpt4o_ja.run(args)


if __name__ == "__main__":
    main()
