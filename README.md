# Distributed RAG Pipeline on Apache Spark

Passage retrieval pipeline for MS MARCO using TF-IDF, BM25, Word2Vec, and hybrid fusion.
Built with Scala + Spark MLlib, packaged with Maven.

## Requirements

- Java 11+
- Apache Spark 3.5.1 (`spark-submit` on PATH)
- Maven 3.6+

## Build

```bash
mvn clean package -q
# produces target/rag-project-assembly.jar
```

## Data Setup

Download MS MARCO from https://microsoft.github.io/msmarco/ and place:
```
data/msmarco/collection.subset.tsv   # 200k passages
data/msmarco/queries.train.tsv
data/msmarco/qrels.train.subset.tsv
```

Convert to Parquet (run once):
```bash
spark-submit --master 'local[*]' --class core.DataLoader \
  target/rag-project-assembly.jar \
  --passages data/msmarco/collection.subset.tsv \
  --queries  data/msmarco/queries.train.tsv \
  --qrels    data/msmarco/qrels.train.subset.tsv \
  --output   data/msmarco/parquet
```

## Build Embeddings

```bash
# TF-IDF
spark-submit --master 'local[*]' --driver-memory 8g --class app.BuildEmbeddings \
  target/rag-project-assembly.jar \
  --method tfidf --input data/msmarco/parquet/passages --output embeddings/tfidf --model models

# BM25
spark-submit --master 'local[*]' --driver-memory 8g --class app.BuildEmbeddings \
  target/rag-project-assembly.jar \
  --method bm25 --input data/msmarco/parquet/passages --output embeddings/bm25 --model models

# Word2Vec
spark-submit --master 'local[*]' --driver-memory 8g --class app.BuildEmbeddings \
  target/rag-project-assembly.jar \
  --method word2vec --input data/msmarco/parquet/passages --output embeddings/word2vec --model models

# Word2Vec-SIF (reuses Word2Vec model if already built)
spark-submit --master 'local[*]' --driver-memory 8g --class app.BuildEmbeddings \
  target/rag-project-assembly.jar \
  --method word2vec-sif --input data/msmarco/parquet/passages --output embeddings/word2vec-sif --model models
```

## Evaluate

```bash
spark-submit --master 'local[*]' --driver-memory 8g --class app.EvalApp \
  target/rag-project-assembly.jar \
  --embedding bm25 --k 10 \
  --embedding-path embeddings/bm25 --model-path models \
  --queries data/msmarco/queries.train.tsv \
  --qrels   data/msmarco/qrels.train.subset.tsv \
  --output  results/bm25-k10.csv

# Hybrid (BM25 sparse + Word2Vec dense)
spark-submit --master 'local[*]' --driver-memory 8g --class app.EvalApp \
  target/rag-project-assembly.jar \
  --embedding hybrid --sparse-embedding bm25 --dense-embedding word2vec \
  --fusion linear --alpha 0.85 --k 10 \
  --embedding-path embeddings/bm25  --model-path models \
  --dense-path     embeddings/word2vec --dense-model models \
  --queries data/msmarco/queries.train.tsv \
  --qrels   data/msmarco/qrels.train.subset.tsv \
  --output  results/hybrid-k10.csv
```

Replace `--embedding` with `tfidf`, `word2vec`, or `word2vec-sif` for other methods.

## Interactive Search

```bash
spark-submit --master 'local[*]' --driver-memory 8g --class app.SearchApp \
  target/rag-project-assembly.jar \
  --embedding bm25 --index embeddings/bm25 --model models --k 10
```

Type a query at the prompt. Use `quit` to exit.

## Run Full Pipeline (Python)

Runs build → data load → all embeddings → all eval configurations automatically:

```bash
python pipeline.py                       # local mode
python pipeline.py --skip-build          # skip mvn package
python pipeline.py --skip-build --skip-data  # skip build and data load
python pipeline.py --master yarn         # YARN cluster
```

Output is saved to `results/pipeline_<timestamp>.log`.

## Supported Options

| Flag | Values | Default |
|------|--------|---------|
| `--embedding` | `tfidf`, `bm25`, `word2vec`, `word2vec-sif`, `hybrid` | `tfidf` |
| `--fusion` | `linear`, `rrf` | `linear` |
| `--alpha` | 0.0 – 1.0 | `0.5` |
| `--k` | any int | `10` |
| `--max-queries` | any int | all |
