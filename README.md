# MIND 15-field Recommendation

Kaggle notebook running full pipeline (reproduce version, tf): https://www.kaggle.com/code/vhutrdng/it4613-pipeline

Pipeline recommendation cho MIND-large, dùng 15 field title, abstract và
metadata, không dùng body.

Notebook Kaggle mẫu:

```text
deployment/kaggle/notebook/[IT4613] pipeline.ipynb
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

Bundle dùng compact typed arrays/memmap, không tạo SQLite candidate database.
Impression/candidate/history được lưu bằng mảng số và offsets; news feature lưu
một lần theo DID. Bundle còn chứa vocab, word/entity matrices, train-DID
membership và manifest SHA-256. Mục tiêu kích thước là khoảng 3–8 GB thay vì
hàng chục GB. Save Version rồi tạo Kaggle Dataset từ thư mục này để tái sử dụng.

Bundle chưa qua neural encoder, nên vẫn dùng được khi đổi attention/DIN/MLP
nhưng giữ nguyên tokenizer, vocab, input lengths và field schema.

### 2. Train một model và tạo submission

```bash
bash deployment/kaggle/run.sh \
  --mode single \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --partition 0 \
  --effective-batch 1024 \
  --batch-size 64 \
  --replicas 2 \
  --precision mixed_float16 \
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

# Predict from the trained checkpoint and create the ZIP
bash deployment/kaggle/run.sh \
  --mode submit \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --run-dir /kaggle/working/mind15/recommendation_runs/partition-0 \
  --predict-batch-size 128 \
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

Để predict trên Kaggle từ run đã upload, truyền `--hf-folder` thay cho
`--run-dir`:

```bash
bash deployment/kaggle/run.sh \
  --mode submit \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --partition 0 \
  --hf-repo vohuutridung/IT4613-HGY \
  --hf-folder experiments/partition-0 \
  --work-root /kaggle/working/mind15
```

Lệnh tải metadata và checkpoint mới nhất của run vào `<work-root>/hf-cache`,
sau đó predict và tạo ZIP như khi dùng run local. Vẫn cần prepared bundle
tương ứng với checkpoint. Repo private cần `HF_TOKEN` có quyền đọc. Dùng
`--hf-revision` để chọn branch, tag hoặc commit (mặc định `main`). Thêm
`--predict-max-candidates 1000000` nếu chỉ muốn chạy smoke test.

Để predict ensemble từ Hugging Face, truyền thư mục cha chứa đủ
`partition-0` đến `partition-4`:

```bash
bash deployment/kaggle/run.sh \
  --mode ensemble-submit \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --hf-repo vohuutridung/IT4613-HGY \
  --hf-folder tf_2011/ \
  --predict-batch-size 8192 \
  --predict-replicas 2 \
  --work-root /kaggle/working/mind15
```

Runner lấy checkpoint mới nhất của mỗi partition từ cùng một revision,
predict lần lượt rồi gộp kết quả thành `submissions/ensemble-submission.zip`.

## Mode và tham số

Các mode:

- `prepare`: preprocess raw MIND.
- `train`: train một partition.
- `benchmark`: so sánh FP32, FP16 và FP16 + XLA trên các run độc lập.
- `single`: train một partition rồi tạo submission.
- `train-ensemble`: train partitions 0 đến 4.
- `submit`: chạy dự đoán từ checkpoint của một model và tạo ZIP submission.
- `ensemble-submit`: tạo ensemble submission.
- `push-model`: upload model lên Hugging Face.

Tham số chính:

- `--data-root`: raw MIND root.
- `--prepared-root`: prepared bundle.
- `--work-root`: output root, mặc định `/kaggle/working/mind15`.
- `--partition`: partition `0..4`.
- `--effective-batch`: effective batch size.
- `--batch-size` (alias `--microbatch`): microbatch **trên mỗi GPU**, mặc định `64`.
- `--predict-batch-size`: batch dự đoán **tổng trên các GPU**, mặc định `128`. Ví dụ batch `512` với hai GPU chia thành khoảng `256` candidates/GPU.
- `--predict-replicas`: số GPU cùng dự đoán từ một checkpoint, mặc định `1`. Chọn `2` để dùng cả hai T4. Mỗi GPU có model và graph riêng, hai worker chạy đồng thời rồi ghép probabilities đúng thứ tự đầu vào. `--replicas` vẫn điều khiển training.
- `--predict-max-candidates`: giới hạn số candidates mỗi model để smoke test predict, ví dụ `500000`. Dùng được với `--mode submit`, `single`, `ensemble-submit`. Kết quả ghi vào `predictions/model-0-test-smoke.csv.gz` (index tương ứng cho ensemble) rồi kết thúc trước bước tạo ZIP. Có thể dùng checkpoint smoke. Bỏ cờ này để chạy toàn bộ tập test và tạo submission.
- `--replicas`: số GPU cùng train một model, mặc định `2`; thiếu GPU sẽ báo lỗi.
- `--precision`: `mixed_float16` (mặc định) hoặc `float32`.
- `--prefetch-batches`: số batch CPU chuẩn bị trước, mặc định `2`; `0` để tắt.
- `--checkpoint-every`: lưu checkpoint mỗi N optimizer updates, mặc định `250`.
- `--train-step-xla` / `--no-train-step-xla`: thử XLA cho forward/backward và accumulation; mặc định tắt.
- `--seed`: random seed.
- `--max-updates`: giới hạn update **và thay đổi LR horizon** để chạy thử.
- `--max-impressions`: giới hạn số impressions train trước khi lấy candidates theo partition. Dùng cùng `--max-updates 5` để smoke test nhanh, gồm cả bước đếm partition. Số impressions khác số candidate examples.
- `--stop-after-updates-this-run`: dừng sau N updates trong phiên, giữ nguyên LR horizon của cả epoch.
- `--benchmark-updates`: số updates mỗi cấu hình khi `--mode benchmark`, mặc định `30`.
- `--resume-run`: checkpoint run cần resume.
- `--run-dir`: model run dùng để submit/upload.
- `--output`: đường dẫn submission ZIP.
- `--hf-repo`, `--hf-folder`: đích upload với `push-model`, thư mục một run với `submit`, hoặc thư mục cha của 5 run với `ensemble-submit`.
- `--hf-revision`: branch, tag hoặc commit trên Hugging Face dùng với `submit` và `ensemble-submit`.
- `--allow-partial`: cho phép dùng partial run.
- `--allow-cpu`: không bắt buộc GPU.
- `--dry-run`: chỉ in kế hoạch.

Smoke test predict song song trên hai T4:

```bash
bash deployment/kaggle/run.sh \
  --mode submit \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --partition 0 \
  --predict-replicas 2 \
  --predict-batch-size 512 \
  --predict-max-candidates 500000 \
  --work-root /kaggle/working/mind15
```

Cho phép nhìn thấy cả hai GPU (`CUDA_VISIBLE_DEVICES=0,1`, mặc định của `run.sh`).
Thiếu GPU được báo lỗi trước khi đọc checkpoint. Bỏ `--predict-max-candidates` để
dự đoán toàn bộ tập test và tạo ZIP. Batch cuối lẻ được chia thành các phần có
kích thước chênh nhau tối đa một candidate, và mọi candidate giữ nguyên vị trí
trong CSV. Log `[predict-memory]` và `inference_execution` có thông tin từng GPU.
Checkpoint từ phiên bản eager và graph trước đây dùng được khi chỉ khác các
phiên bản execution/CLI đã biết, các hash model/data/protocol còn lại vẫn phải khớp.

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
- Effective batch: `1024`; per-GPU microbatch: `64`; GPU replicas: `2`.
- Accumulation tự suy ra: `1024 / (64 × 2) = 8`; không có flag grad accumulation.
- Effective batch phải chia hết cho `batch-size × replicas`.
- Mixed precision FP16, FP32 loss/attention normalization/gradient accumulation và dynamic loss scaling.
- Ordered CPU prefetch `2` batches; vectorized memmap feature/history gathering.
- Graph loop tích lũy gradient trên từng GPU; sparse embedding gradients được gộp theo index trước khi đồng bộ ở optimizer update.
- Batch cuối được padding với trọng số loss bằng 0 và chuẩn hóa theo số candidates thực tế.
- Negative partitions: `5`.
- Default partition: `0`.
- Seed: `20261110`.
- Checkpoint mỗi `250` updates, giữ `2` checkpoint gần nhất; luôn lưu khi kết thúc phiên.
- Loss scale và bộ đếm scale được lưu cùng model/Adam/candidate cursor. Overflow FP16 sẽ tính lại cùng batch với scale thấp hơn, không tăng Adam/cursor/LR step.

Các flags hiệu năng và source fingerprint được khóa trong `run_config.json`.
Khi chuyển từ implementation cũ hoặc đổi batch/replicas/precision, dùng work-root mới;
checkpoint cũ không được resume trực tiếp. Prepared bundle hiện tại vẫn dùng lại được.

### Benchmark FP32 / FP16 / XLA trên cùng hai GPU

```bash
bash deployment/kaggle/run.sh \
  --mode benchmark \
  --prepared-root /kaggle/input/CHANGE_ME_PREPARED/prepared \
  --effective-batch 1024 --batch-size 64 --replicas 2 \
  --benchmark-updates 30 \
  --work-root /kaggle/working/mind15-benchmark
```

Mỗi case dùng một run mới, cùng seed/partition và LR horizon của cả epoch.
Kết quả nằm trong `benchmarks/benchmark_summary.json`; trường
`steady_state_candidates_per_second` bỏ qua 5 updates đầu để loại thời gian trace/compile.
Số đo này gồm chờ batch, train step và checkpoint định kỳ, không gồm khởi tạo/đếm
partition hoặc checkpoint cuối phiên. Adam JIT mặc định tắt; flag XLA chỉ compile graph tính và tích lũy gradient.
Lỗi XLA được ghi rõ; không tự bật XLA cho train chính. Không có submission từ benchmark. Dùng work-root mới khi benchmark lại.

CPU smoke test: thêm `--replicas 1 --precision float32 --allow-cpu`.
Multi-GPU runtime checks (trong môi trường TF 2.15.1 của Kaggle):

```bash
MIND_TEST_GPUS=1 /kaggle/working/mind15/python311-tf215/bin/python \
  -m unittest discover -s tests -p test_distributed_training.py
```

Nếu không đặt `MIND_TEST_GPUS`, test dùng hai logical CPU. Đặt thêm
`MIND_TEST_XLA=1` để kiểm tra khả năng compile sparse train step.

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
