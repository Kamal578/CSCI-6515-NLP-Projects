# Dataset Profile

## Source
- Dataset: ESG Sustainability Reports of S&P 500 Companies
- Link: https://www.kaggle.com/datasets/jaidityachopra/esg-sustainability-reports-of-s-and-p-500-companies/data

## Local Snapshot
- Profiling date: 2026-04-22
- Input file: `data/preprocessed_content.csv`
- Generated document directory: `data/raw_txt/`

## Size Summary
- Rows/documents in CSV: 866
- Non-empty `preprocessed_content` rows: 866
- Generated TXT files: 866
- Total words (whitespace tokenization): 10,361,942
- Average words per document: 11,965.29

## Indexing Summary
- Chunking bounds: strict (201, 499)
- Target chunk size: 420
- Documents skipped during chunking: 3
- Stored chunks: 24,687
- Chunk token stats:
  - Min: 215
  - Max: 488
  - Mean: 419.73
  - Median: 420

## Notes
- Short/unsplittable documents outside strict bounds are skipped and reported by the indexing script.
- `vector_db/` is generated local state and is excluded from version control.
