For your **Armenian Wikipedia → Apache Spark → Llama LoRA** project, you need **one main Wikipedia dump file**. Everything else is either optional or unnecessary.

## 1. Which files should you download?

Open the official directory:

[Armenian Wikipedia — latest dumps](https://dumps.wikimedia.org/hywiki/latest/?utm_source=chatgpt.com)

Look for these files:

| File | Download? | Purpose |
|---|---|---|
| `hywiki-latest-pages-articles-multistream.xml.bz2` | **YES** | Wikipedia article content |
| `hywiki-latest-pages-articles-multistream-index.txt.bz2` | Optional | Index for random access to articles |
| `hywiki-latest-pages-articles.xml.bz2` | Alternative | Same article content without multistream organization |
| `hywiki-latest-pages-meta-current.xml.bz2` | No | Current pages from all namespaces |
| `hywiki-latest-pages-meta-history.xml.bz2` | No | Full revision history |
| `hywiki-latest-stub-articles.xml.gz` | No | Metadata without article text |
| `hywiki-latest-page.sql.gz` | No | Database metadata |
| `hywiki-latest-categorylinks.sql.gz` | Optional | Category information for later experiments |
| `hywiki-latest-md5sums.txt` | Recommended | Verify download integrity, if provided |

**My recommendation:** Download the multistream XML file and the checksum file.

The multistream dump contains the current article revisions and is sufficient for language-model training. You don't need the separate index unless you want random access to individual compressed streams. [Meta-Wiki](https://meta.wikimedia.org/wiki/Data_dumps/Dump_format?utm_source=chatgpt.com)

### Download commands (macOS/Linux)

```bash
mkdir -p armenian-llama/data/raw
cd armenian-llama/data/raw

curl -fL -C - -O \
  https://dumps.wikimedia.org/hywiki/latest/hywiki-latest-pages-articles-multistream.xml.bz2
```

For reproducible experiments, record the actual dump date and preferably use a dated directory instead of `latest`.

You can inspect the file without fully decompressing it:

```bash
bzip2 -dc hywiki-latest-pages-articles-multistream.xml.bz2 | head -c 2000
```

---

# 2. Complete processing pipeline

I recommend the following architecture:

```text
                  ARMENIAN WIKIPEDIA
                          │
                          ▼
       pages-articles-multistream.xml.bz2
                          │
                          ▼
                Wikipedia Extractor
                          │
                          ▼
                Article JSON files
                          │
                          ▼
             ┌──────────────────────┐
             │ Apache Spark / Scala │
             ├──────────────────────┤
             │ Unicode normalization│
             │ Armenian filtering   │
             │ Quality filtering    │
             │ Deduplication        │
             │ Corpus statistics    │
             │ Train/validation/test│
             └──────────┬───────────┘
                        │
                        ▼
                 Clean Parquet
                        │
                        ▼
                JSONL export
                        │
                        ▼
                Llama tokenizer
                        │
                        ▼
                 Token packing
                        │
                        ▼
              Llama + LoRA training
                        │
                        ▼
              Armenian Llama adapter
```

There are **four major processing stages**.

## 3. Stage 1 — Extract Wikipedia articles

The downloaded XML contains Wikitext, not clean natural-language text.

For example:

```xml
<page>
  <title>Երևան</title>
  <ns>0</ns>
  <id>12345</id>
  <revision>
    <text xml:space="preserve">
      '''Երևան'''ը [[Հայաստան]]ի մայրաքաղաքն է։
    </text>
  </revision>
</page>
```

We want:

```json
{
  "id": "12345",
  "title": "Երևան",
  "text": "Երևանը Հայաստանի մայրաքաղաքն է։"
}
```

### Install WikiExtractor

[WikiExtractor GitHub repository](https://github.com/attardi/wikiextractor?utm_source=chatgpt.com)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install wikiextractor
```

Run:

```bash
python -m wikiextractor.WikiExtractor \
    --json \
    --processes 8 \
    -o data/extracted \
    data/raw/hywiki-latest-pages-articles-multistream.xml.bz2
```

This produces directories containing JSONL article records.

**Important:** WikiExtractor is an older project. Test it with your Python version before running the full extraction. Also inspect extracted Armenian text manually: template expansion, links, and tables may not be rendered perfectly.

---

# 4. Stage 2 — Process articles with Spark

Now we use Apache Spark.

I recommend **Scala for Spark processing** and Python for model training.

### Read the extracted JSON

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("ArmenianWikipedia")
  .master("local[*]")
  .getOrCreate()

import spark.implicits._

val wiki = spark.read
  .json("data/extracted/*/*")

wiki.printSchema()
```

Expected columns:

```text
root
 |-- id: string
 |-- title: string
 |-- url: string
 |-- text: string
```

### Clean the data

```scala
val cleaned = wiki
  .filter(col("text").isNotNull)
  .withColumn(
    "text",
    trim(regexp_replace(col("text"), "\\s+", " "))
  )
  .filter(length(col("text")) > 200)
  .withColumn("char_count", length(col("text")))
  .withColumn(
    "word_count",
    size(split(col("text"), "\\s+"))
  )
```

For Armenian-language filtering, we can calculate the proportion of Armenian letters.

```scala
val withLanguageRatio = cleaned
  .withColumn(
    "armenian_letters",
    length(
      regexp_replace(
        col("text"),
        "[^\\u0531-\\u0556\\u0561-\\u0587]",
        ""
      )
    )
  )
  .withColumn(
    "all_letters",
    length(
      regexp_replace(
        col("text"),
        "[^\\p{L}]",
        ""
      )
    )
  )
  .withColumn(
    "armenian_ratio",
    col("armenian_letters") / greatest(col("all_letters"), lit(1))
  )
  .filter(col("armenian_ratio") > 0.6)
```

The threshold `0.6` is an experimental starting point, not a linguistic rule.

You should measure how many articles it removes and manually inspect a sample of rejected documents.

### Deduplicate

```scala
val deduplicated = withLanguageRatio
  .withColumn("content_hash", sha2(col("text"), 256))
  .dropDuplicates("content_hash")
```

This removes exact duplicates.

Later, students can implement near-duplicate detection using Spark MLlib MinHashLSH.

### Save as Parquet

```scala
deduplicated.write
  .mode("overwrite")
  .parquet("data/processed/hywiki")
```

This is the canonical dataset.

---

# 5. Stage 3 — Prepare Llama training data

For **Armenian-language adaptation**, we want causal language-model training.

We do not need questions and answers.

The required data is simply:

```json
{"text":"Երևանը Հայաստանի Հանրապետության մայրաքաղաքն է։"}
{"text":"Հայաստանը գտնվում է Հարավային Կովկասում։"}
{"text":"Սևանա լիճը Հայաստանի խոշորագույն լիճն է։"}
```

Each line is a separate JSON object.

### Split the dataset

Use a deterministic split based on article IDs:

```scala
val withSplit = deduplicated.withColumn(
  "split_bucket",
  pmod(xxhash64(col("id")), lit(100))
)

val train = withSplit.filter(col("split_bucket") < 90)

val validation = withSplit.filter(
  col("split_bucket") >= 90 &&
  col("split_bucket") < 95
)

val test = withSplit.filter(col("split_bucket") >= 95)
```

This gives approximately:

- 90% training
- 5% validation
- 5% testing

Split before tokenization to prevent article-level leakage.

### Export JSONL

```scala
train.select("text")
  .write.mode("overwrite")
  .json("data/training/train")

validation.select("text")
  .write.mode("overwrite")
  .json("data/training/validation")

test.select("text")
  .write.mode("overwrite")
  .json("data/training/test")
```

Spark produces JSONL part files.

For MLX-LM, it is convenient to assemble the output into:

```text
data/training/
    train.jsonl
    valid.jsonl
    test.jsonl
```

Avoid using `coalesce(1)` for very large corpora, because it creates a single-task bottleneck. For Wikipedia-sized experiments, merging exported shards afterward is also an option.

---

# 6. Stage 4 — Fine-tune Llama with LoRA

For your **MacBook Pro M1 Max, 32 GB**, I recommend Apple's MLX framework.

[MLX-LM GitHub](https://github.com/ml-explore/mlx-lm?utm_source=chatgpt.com)

It is designed for Apple Silicon and supports LoRA training.

### Install

```bash
pip install mlx-lm
```

### Choose a model

For the first experiment, use a small base model, such as:

[Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B?utm_source=chatgpt.com)

This is a **base model**, not an instruction-tuned model.

You will need to accept Meta's model license and obtain access to the checkpoint.

Llama 3.2 1B is a reasonable starting size for your machine.

### Prepare the training files

```text
data/training/
    train.jsonl
    valid.jsonl
    test.jsonl
```

Each record:

```json
{"text":"Հայաստանի Հանրապետությունը գտնվում է Հարավային Կովկասում։"}
```

### Start LoRA training

A typical MLX-LM command is:

```bash
mlx_lm.lora \
  --model meta-llama/Llama-3.2-1B \
  --train \
  --data data/training \
  --batch-size 1 \
  --num-layers 16 \
  --iters 1000 \
  --learning-rate 1e-4 \
  --adapter-path adapters/armenian-lora
```

Depending on the installed MLX-LM version, model conversion, configuration options, and model access requirements may differ. Check:

```bash
mlx_lm.lora --help
```

The command above is a starting configuration, not an optimized training recipe.

### What happens internally?

```text
Armenian text
      ↓
Llama tokenizer
      ↓
Token IDs
      ↓
Llama decoder
      ↓
Next-token predictions
      ↓
Cross-entropy loss
      ↓
Backpropagation
      ↓
Update LoRA adapters only
```

The original Llama weights remain frozen.

For Wikipedia language adaptation, this is **LoRA-based continued pretraining**, not instruction fine-tuning.

---

# 7. What exactly does LoRA learn?

Consider:

```text
Հայաստանի մայրաքաղաքը Երևանն է։
```

The model is trained to predict each subsequent token based on preceding tokens.

Conceptually:

| Context | Expected continuation |
|---|---|
| Հայաստանի | մայրաքաղաքը |
| Հայաստանի մայրաքաղաքը | Երևանն |
| Հայաստանի մայրաքաղաքը Երևանն | է |
| Հայաստանի մայրաքաղաքը Երևանն է | ։ |

The actual training targets are tokenizer IDs, not necessarily whole Armenian words.

LoRA updates low-rank matrices attached to selected Transformer layers.

Mathematically:

\[
W'=W+\frac{\alpha}{r}BA
\]

where:

- \(W\) is the frozen pretrained weight matrix.
- \(A,B\) are trainable low-rank matrices.
- \(r\) is the LoRA rank.
- \(\alpha\) controls the adapter scaling.

The training process minimizes causal language-model loss while modifying only those adapter parameters.

---

# 8. Evaluate the adapted model

I recommend three comparisons:

| Model | Armenian perplexity | Generation quality |
|---|---|---|
| Original Llama | Baseline | Baseline |
| Llama + Armenian LoRA | Measure | Measure |
| Llama + full continued pretraining | Optional | Optional |

For perplexity, use the held-out Wikipedia test articles.

\[
\text{Perplexity}=\exp(\text{average token-level cross-entropy})
\]

Use the same tokenizer and evaluation data for both models.

Also test generation with Armenian prompts such as:

```text
Հայաստանի մայրաքաղաքը
```

```text
Հայկական գրականության պատմության մեջ
```

```text
Սևանա լիճը գտնվում է
```

A lower perplexity does not automatically guarantee better conversational Armenian. For that, you would eventually need instruction fine-tuning and human evaluation.

---

## 9. Final project structure

```text
armenian-llama/
│
├── data/
│   ├── raw/
│   │   └── hywiki-latest-pages-articles-multistream.xml.bz2
│   │
│   ├── extracted/
│   │   └── ...
│   │
│   ├── processed/
│   │   └── hywiki/
│   │
│   └── training/
│       ├── train.jsonl
│       ├── valid.jsonl
│       └── test.jsonl
│
├── spark/
│   ├── WikipediaProcessing.scala
│   ├── CorpusStatistics.scala
│   └── DatasetExport.scala
│
├── training/
│   ├── tokenizer_analysis.py
│   ├── train_lora.sh
│   └── evaluate.py
│
└── adapters/
    └── armenian-lora/
```

### One important limitation

Armenian Wikipedia is a good starting corpus, but it may be too small to substantially improve a 1B-parameter model's Armenian fluency.

Before training, calculate the total number of **Llama tokens**, not just words or bytes. This will determine whether you have enough training material and help estimate the number of useful training iterations.

**My recommendation:** Start with the article multistream dump, build the Spark pipeline, and train a Llama 3.2 1B LoRA adapter. Treat this as a reproducible baseline experiment, then evaluate whether additional Armenian corpora are needed.