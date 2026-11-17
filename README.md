# MIND 15-field Recommendation

Pipeline recommendation cho MIND-large, dùng 15 field title, abstract và
metadata, không dùng body.

Notebook Kaggle mẫu:

```text
deployment/kaggle/notebook/mind_15field_training.ipynb
```

## Chạy trên Kaggle

Chọn GPU, bật Internet và clone repository:

```bash
git clone --branch main --depth 1 \
  https://github.com/vohuutridung/IT4613.git \
  /kaggle/working/IT4613

cd /kaggle/working/IT4613
```

### 1. Preprocess một lần

Raw data cần có ba thư mục:

```text
<data-root>/
├── MINDlarge_train/
├── MINDlarge_dev/
└── MINDlarge_test/
```

Mỗi thư mục cần `news.tsv`, `behaviors.tsv` và `entity_embedding.vec`.

```bash
bash deployment/kaggle/run.sh \
  --mode prepare \
  --data-root /kaggle/input/CHANGE_ME_RAW_MIND \
  --work-root /kaggle/working/mind15
```

Prepared bundle được tạo tại:

```text
/kaggle/working/mind15/prepared
```

Bundle gồm SQLite, vocab, word/entity matrices, train-DID membership và
manifest SHA-256. Save Version rồi tạo Kaggle Dataset từ thư mục này để tái sử
dụng. Bundle chưa qua neural encoder, nên vẫn dùng được khi đổi attention/DIN/
MLP nhưng giữ nguyên tokenizer, vocab, input lengths và field schema.

### 2. Train một model và tạo submission

```bash
bash deployment/kaggle/run.sh \
  --mode single \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --partition 0 \
  --work-root /kaggle/working/mind15
```

Output:

```text
/kaggle/working/mind15/recommendation_runs/partition-0/
/kaggle/working/mind15/submissions/partition-0-submission.zip
```

ZIP chỉ chứa `prediction.txt` theo format CodaLab:

```text
ImpressionID [Rank1,Rank2,...,RankN]
```

### 3. Train và submit riêng

```bash
# Train
bash deployment/kaggle/run.sh \
  --mode train \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --partition 0 \
  --work-root /kaggle/working/mind15

# Submit
bash deployment/kaggle/run.sh \
  --mode submit \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --run-dir /kaggle/working/mind15/recommendation_runs/partition-0 \
  --output /kaggle/working/mind15/submissions/partition-0.zip \
  --work-root /kaggle/working/mind15
```

### 4. Resume

```bash
bash deployment/kaggle/run.sh \
  --mode train \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --partition 0 \
  --resume-run /kaggle/input/CHANGE_ME_OLD_RUN/partition-0 \
  --work-root /kaggle/working/mind15
```

### 5. Ensemble

```bash
# Train partitions 0..4
bash deployment/kaggle/run.sh \
  --mode train-ensemble \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --work-root /kaggle/working/mind15

# Predict, mean probability và tạo ZIP
bash deployment/kaggle/run.sh \
  --mode ensemble-submit \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --work-root /kaggle/working/mind15
```

Output:

```text
/kaggle/working/mind15/submissions/ensemble-submission.zip
```

### 6. Upload model lên Hugging Face

Tạo Kaggle Secret `HF_TOKEN`, sau đó nạp vào environment:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```bash
bash deployment/kaggle/run.sh \
  --mode push-model \
  --run-dir /kaggle/working/mind15/recommendation_runs/partition-0 \
  --hf-repo vohuutridung/IT4613-HGY \
  --hf-folder experiments/partition-0 \
  --work-root /kaggle/working/mind15
```

Chỉ checkpoint và run metadata được upload.

## Mode và tham số

Các mode:

- `prepare`: preprocess raw MIND.
- `train`: train một partition.
- `single`: train một partition rồi tạo submission.
- `train-ensemble`: train partitions 0 đến 4.
- `submit`: tạo submission từ một model.
- `ensemble-submit`: tạo ensemble submission.
- `push-model`: upload model lên Hugging Face.

Tham số chính:

- `--data-root`: raw MIND root.
- `--prepared-root`: prepared bundle.
- `--work-root`: output root, mặc định `/kaggle/working/mind15`.
- `--partition`: partition `0..4`.
- `--effective-batch`: effective batch size.
- `--microbatch`: microbatch size.
- `--seed`: random seed.
- `--max-updates`: giới hạn update để chạy thử.
- `--resume-run`: checkpoint run cần resume.
- `--run-dir`: model run dùng để submit/upload.
- `--output`: đường dẫn submission ZIP.
- `--hf-repo`, `--hf-folder`: đích upload Hugging Face.
- `--allow-partial`: cho phép dùng partial run.
- `--allow-cpu`: không bắt buộc GPU.
- `--dry-run`: chỉ in kế hoạch.

## Config mặc định

Config đầy đủ:

```text
configs/mind_no_body_15field.json
```

Training:

- Fields: `15 × 100`.
- MLP: `1024 → 512 → 256 → 1`.
- Loss: pointwise sigmoid BCE.
- Optimizer: Adam.
- Peak/min learning rate: `1e-3` / `1e-6`.
- Warmup: `10%`.
- Effective batch / microbatch: `1024` / `16`.
- Negative partitions: `5`.
- Default partition: `0`.
- Seed: `20261110`.
- Checkpoint mỗi `25` updates, giữ `2` checkpoint gần nhất.

Input limits:

- Title / abstract: `30` / `50` tokens.
- History: `200` news.
- History title / abstract: `50` / `10`.
- Title / abstract entities: `9` / `30`.

Masking:

- Dev mask seed: `1024`.
- Unseen dev DID mask ratio: `0.92`.
- Unseen test DID được map thành `UNK`.

## Logging

Live progress hiển thị stage, loss, learning rate, throughput, ETA và
checkpoint. Log được lưu tại:

```text
<work-root>/pipeline.log
<work-root>/pipeline-events.jsonl
<run-dir>/training_progress.json
```

## Kiểm tra

```bash
python3 -m unittest tests/test_kaggle_workflow.py
bash -n deployment/kaggle/run.sh
python3 deployment/kaggle/runner.py --help
```

Chi tiết kỹ thuật và đối chiếu upstream nằm trong `docs/`.
