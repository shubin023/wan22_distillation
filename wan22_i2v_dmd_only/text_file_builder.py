import argparse
import pandas as pd
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build individual text files from a CSV.")
    parser.add_argument("--csv_path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to write the text files.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for col1, col2 in zip(df['path'], df['i2v_fusion_caption']):
        file_name = os.path.splitext(col1.split("/")[-1])[0] + ".txt"
        out_path = Path(args.output_dir) / (file_name)
        out_path.write_text(str(col2), encoding="utf-8")


if __name__ == "__main__":
    main()
