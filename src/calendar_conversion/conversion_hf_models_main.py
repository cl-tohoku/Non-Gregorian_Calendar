import argparse

import conversion_hf_models_en
import conversion_hf_models_ja
import conversion_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    args = parser.parse_args()

    conversion_hf_models_en.run(args)
    conversion_hf_models_ja.run(args)
    conversion_plot.run(args)

if __name__ == "__main__":
    main()
