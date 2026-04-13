#!/usr/bin/env python3
"""
pipeline.py — full RAG pipeline: build jar → load data → build embeddings → evaluate.

Spark log noise is filtered; only application output is written to the log file.

Usage:
    python pipeline.py                      # local mode (default)
    python pipeline.py --master yarn        # YARN cluster
    python pipeline.py --skip-build         # skip mvn package
    python pipeline.py --skip-data          # skip DataLoader (parquet already exists)
    python pipeline.py --driver-memory 16g  # override driver memory
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

# ── Spark log noise filter ────────────────────────────────────────────────────

SPARK_LOG   = re.compile(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|ERROR|DEBUG)\b")
SPARK_STAGE = re.compile(r"^\[Stage \d+")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

def is_noise(line):
    clean = ANSI_ESCAPE.sub("", line)
    return bool(SPARK_LOG.match(clean) or SPARK_STAGE.match(clean))

# ── Runner ────────────────────────────────────────────────────────────────────

def run(label, cmd, log):
    """Run cmd, stream filtered output to stdout and log file."""
    separator = f"\n{'=' * 60}\n{label}\n{'=' * 60}"
    print(separator)
    log.write(separator + "\n")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    for raw_line in proc.stdout:
        line = ANSI_ESCAPE.sub("", raw_line).rstrip()
        if line and not is_noise(line):
            print(line)
            log.write(line + "\n")
            log.flush()

    proc.wait()
    if proc.returncode != 0:
        msg = f"\n[FAILED] '{label}' exited with code {proc.returncode}"
        print(msg, file=sys.stderr)
        log.write(msg + "\n")
        sys.exit(proc.returncode)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the full RAG pipeline")
    parser.add_argument("--master",        default="local[*]",  help="Spark master URL (default: local[*])")
    parser.add_argument("--driver-memory", default="8g",        help="Spark driver memory (default: 8g)")
    parser.add_argument("--skip-build",    action="store_true", help="Skip mvn package")
    parser.add_argument("--skip-data",     action="store_true", help="Skip DataLoader (parquet already built)")
    parser.add_argument("--skip-embed",    action="store_true", help="Skip embedding build step")
    parser.add_argument("--output",        default="",          help="Log file path (default: results/pipeline_<timestamp>.log)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    log_path = args.output or f"results/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    jar    = "target/rag-project-assembly.jar"
    master = args.master
    mem    = args.driver_memory

    def spark(cls, *spark_args):
        """Build a spark-submit command list."""
        return ["spark-submit", "--master", master, "--driver-memory", mem,
                "--class", cls, jar, *spark_args]

    with open(log_path, "w") as log:
        header = f"RAG Pipeline  started={datetime.now().isoformat()}  master={master}\n"
        print(header); log.write(header)

        # 1. Build fat JAR
        if not args.skip_build:
            run("mvn package", ["mvn", "clean", "package", "-q"], log)

        # 2. Load TSV → Parquet
        if not args.skip_data:
            run("DataLoader", spark(
                "core.DataLoader",
                "--passages", "data/msmarco/collection.subset.tsv",
                "--queries",  "data/msmarco/queries.train.tsv",
                "--qrels",    "data/msmarco/qrels.train.subset.tsv",
                "--output",   "data/msmarco/parquet",
            ), log)

        # 3. Build embeddings
        if not args.skip_embed:
            for method, out_dir in [
                ("tfidf",       "embeddings/tfidf"),
                ("bm25",        "embeddings/bm25"),
                ("word2vec",    "embeddings/word2vec"),
                ("word2vec-sif","embeddings/word2vec-sif"),
            ]:
                run(f"BuildEmbeddings [{method}]", spark(
                    "app.BuildEmbeddings",
                    "--method", method,
                    "--input",  "data/msmarco/parquet/passages",
                    "--output", out_dir,
                    "--model",  "models",
                ), log)

        # 4. Evaluate — single methods
        single_evals = [
            ("tfidf",       "embeddings/tfidf",       "models", "results/tfidf-k10.csv"),
            ("bm25",        "embeddings/bm25",        "models", "results/bm25-k10.csv"),
            ("word2vec",    "embeddings/word2vec",    "models", "results/word2vec-k10.csv"),
            ("word2vec-sif","embeddings/word2vec-sif","models", "results/word2vec-sif-k10.csv"),
        ]

        for emb, emb_path, model, out_csv in single_evals:
            run(f"EvalApp [{emb}]", spark(
                "app.EvalApp",
                "--embedding",      emb,
                "--k",              "10",
                "--embedding-path", emb_path,
                "--model-path",     model,
                "--queries",        "data/msmarco/queries.train.tsv",
                "--qrels",          "data/msmarco/qrels.train.subset.tsv",
                "--output",         out_csv,
            ), log)

        # 5. Evaluate — hybrid configurations
        hybrid_evals = [
            {
                "label":   "hybrid [bm25+word2vec linear α=0.5]",
                "sparse":  "bm25",       "sparse_path": "embeddings/bm25",
                "dense":   "word2vec",   "dense_path":  "embeddings/word2vec",
                "fusion":  "linear",     "alpha":       "0.5",
                "output":  "results/hybrid-bm25-word2vec-linear-a0.5-k10.csv",
            },
            {
                "label":   "hybrid [bm25+word2vec linear α=0.85]",
                "sparse":  "bm25",       "sparse_path": "embeddings/bm25",
                "dense":   "word2vec",   "dense_path":  "embeddings/word2vec",
                "fusion":  "linear",     "alpha":       "0.85",
                "output":  "results/hybrid-bm25-word2vec-linear-a0.85-k10.csv",
            },
            {
                "label":   "hybrid [bm25+word2vec rrf]",
                "sparse":  "bm25",       "sparse_path": "embeddings/bm25",
                "dense":   "word2vec",   "dense_path":  "embeddings/word2vec",
                "fusion":  "rrf",        "alpha":       "0.5",
                "output":  "results/hybrid-bm25-word2vec-rrf-k10.csv",
            },
            {
                "label":   "hybrid [bm25+word2vec-sif linear α=0.85]",
                "sparse":  "bm25",       "sparse_path": "embeddings/bm25",
                "dense":   "word2vec-sif","dense_path":  "embeddings/word2vec-sif",
                "fusion":  "linear",     "alpha":       "0.85",
                "output":  "results/hybrid-bm25-word2vec-sif-linear-a0.85-k10.csv",
            },
        ]

        for h in hybrid_evals:
            cmd_args = [
                "--embedding",        "hybrid",
                "--sparse-embedding", h["sparse"],
                "--dense-embedding",  h["dense"],
                "--fusion",           h["fusion"],
                "--alpha",            h["alpha"],
                "--k",                "10",
                "--embedding-path",   h["sparse_path"],
                "--model-path",       "models",
                "--dense-path",       h["dense_path"],
                "--dense-model",      "models",
                "--queries",          "data/msmarco/queries.train.tsv",
                "--qrels",            "data/msmarco/qrels.train.subset.tsv",
                "--output",           h["output"],
            ]
            run(f"EvalApp [{h['label']}]", spark("app.EvalApp", *cmd_args), log)

        footer = f"\nPipeline complete. Log saved to: {log_path}\n"
        print(footer); log.write(footer)


if __name__ == "__main__":
    main()
