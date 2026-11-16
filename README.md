# MIND-large: phân tích đề bài và pipeline tái triển khai 15 field không body

## Tóm tắt

Đây là pipeline tái triển khai giải pháp news recommendation của Huige Cheng cho MIND-large. Phiên bản đang dùng là biến thể **15 field không body**: giữ các field ID, category, entity, title và abstract; bỏ hẳn `cur_body` và `his_body`. Đây không phải implementation nguyên bản 17 field, không được chứng minh tương đương số học với các thư viện riêng `gezi/melt/husky`, và chưa có căn cứ để tuyên bố đạt kết quả cuộc thi.

## 1. Bài toán và protocol cuộc thi

Mỗi impression gồm user, thời điểm, lịch sử tin đã click và một danh sách candidate news. Nhãn click gắn với từng candidate trong impression. Mục tiêu là dự đoán click probability cho từng candidate để xếp hạng chúng trong đúng impression; vì thế metric chính được tổng hợp theo impression, không chỉ trên một bảng candidate phẳng.

Technical Report mô tả bối cảnh bốn tuần dữ liệu huấn luyện để dự đoán tuần thứ năm; tỷ lệ news mới ở test cao, nên nội dung tin và biểu diễn lịch sử quan trọng. Báo cáo yêu cầu mask ngẫu nhiên 92% document IDs ở dev **nếu DID không xuất hiện trong train** để làm dev gần test hơn. Cấu hình hiện hành là `dev_new_did_masked`: membership đến từ DID trong history hoặc candidate của raw `train/behaviors.tsv`, không lấy từ toàn bộ news metadata và không dùng nhãn. Phép rút thăm được khóa theo từng candidate occurrence `(seed, impression_id, position)`; DID đã thấy trong train luôn được giữ. Đây là tái triển khai theo ý định của báo cáo, không phải hành vi executable tương đương: upstream `src/util.py::mask_dids` trong nhánh `valid` đang mask cả tensor DID và không dùng `dids_in_train`, còn cách mask chỉ DID mới nằm trong comment. Báo cáo dùng first 1M instances cho dev; local mặc định đánh giá full dev. Đây là **báo cáo giải pháp cuộc thi**, khác với bài báo giới thiệu dataset MIND. [Huige Cheng, *MIND News Recommendation — Technical Report*, p. 2](https://msnews.github.io/assets/doc/1.pdf)

`source_lock.json` và `source_manifest.json` là hồ sơ khóa snapshot upstream bất biến. Chúng ghi nguồn lịch sử, không phải trạng thái xác minh của từng DB/checkpoint hiện tại. Trạng thái pipeline hiện tại nằm trong cấu hình hoạt động và ma trận đối chiếu dưới `docs/`; DB phải được xác minh theo nội dung khi một lệnh pipeline thực sự chạy.

Báo cáo chọn pointwise training. Click là positive, các candidate còn lại là negative; thay vì dùng toàn bộ negative, tác giả giữ mọi positive và chia negative thành năm partition, chủ yếu chạy partition 0. Báo cáo ghi nhận shuffle giữa impressions phù hợp với per-impression AUC hơn shuffle ngẫu nhiên toàn bộ candidate. Sau này các partition khác được dùng cho ensemble nếu cần. Các kết quả và kết luận trong báo cáo thuộc mô hình của tác giả, không tự động áp dụng cho biến thể không body ở đây. [Technical Report, pp. 2–3](https://msnews.github.io/assets/doc/1.pdf)

## 2. Nguồn chuẩn và cách khóa cấu hình

- Snapshot mô hình: [`chenghuige/mind`, commit `f632bdf7b1136dff142835f9fe69c8ce5f8c166d`](https://github.com/chenghuige/mind/tree/f632bdf7b1136dff142835f9fe69c8ce5f8c166d). Bằng chứng snapshot trong workspace được ghi ở [`source_lock.json`](source_lock.json), [`source_manifest.json`](source_manifest.json) và [`source_provenance.json`](source_provenance.json).
- Cấu hình chạy được truy từ `projects/ai/mind/src/train/v17/body.sh` → `base.sh` → `common.sh`, cùng `src/config.py`, `src/util.py`, `src/model.py`, `prepare/config.py` và `prepare/gen-records.py`.
- Cấu hình thực thi hiện tại đặt trong [`configs/mind_no_body_15field.json`](configs/mind_no_body_15field.json), được nạp qua [`mind_config.py`](mind_config.py). Cấu hình tách preprocessing, encoder, training và execution để tránh rải default trùng nhau.
- Source shell có hai giá trị batch khác nhau: `common.sh` đặt 4096, còn `base.sh` nối thêm 1024; báo cáo nêu batch 1024. Repo hiện chọn effective batch 1024, nhưng không khẳng định đã khôi phục đầy đủ precedence của parser upstream.
- Báo cáo nêu Adam, learning rate tối đa 0.001, warmup 0.1 epoch, triangle schedule, một epoch và một GPU. Shell gọi `bert-adam`, nhưng code snapshot không đủ để xác định implementation cụ thể bên trong dependency ngoài. Pipeline hiện dùng Keras Adam với tham số khai báo tường minh; đây là lựa chọn tái triển khai, không được xem là tương đương `bert-adam`.

## 3. Luồng thực thi hiện tại

### Bước 1 — Raw MIND → vocabulary, mapping và SQLite

Code chính: [`mind_data_pipeline/preprocess.py`](mind_data_pipeline/preprocess.py), cấu hình tại [`mind_data_pipeline/config.py`](mind_data_pipeline/config.py).

1. Đọc `behaviors.tsv` và `news.tsv` của train/dev/test; không trộn split và không lọc candidate theo nhãn.
2. Tách impression ID, user ID, thời điểm, history và candidate tokens. Click label được lưu riêng với candidate; test không nhãn được biểu diễn null cùng trạng thái chưa gán nhãn.
3. Tạo vocabulary/mapping có thứ tự xác định cho UID, DID, category, subcategory, entity, entity type và word. Quy ước local dành PAD ID 0 và UNK ID 1. Đây là mapping tái triển khai; thứ tự ID của `gezi.Vocab` riêng chưa được chứng minh giống nhau.
4. Giữ news content một lần theo news ID, lưu candidate/impression bằng khóa và tham chiếu. History được lưu một lần cho mỗi impression thay vì sao chép toàn bộ history vào mỗi candidate.
5. History được đảo theo hướng xử lý thể hiện trong upstream rồi giới hạn theo profile. Local giữ cách đếm DID đầu tiên trong history đã thấy trong notebook upstream. Local sửa nhánh abstract-history entity để đọc cột abstract đúng; upstream có stale-variable path ở `prepare/gen-records.py`, vì vậy output ở nhánh này khác upstream.
6. News trùng ID được duyệt theo thứ tự train → dev → test; dòng lặp chỉ được bỏ qua nếu toàn bộ raw row giống nhau, nếu khác thì báo lỗi. Chính sách này được ghi trong preprocessing và field contract.

Preprocessor giữ schema Stage 01 kiểu body-v17 để DB đã chuẩn bị còn có thể dùng chung, nhưng loader của biến thể hiện hành không đọc body. Thay đổi code preprocessing có thể đổi fingerprint. Không dùng `--resume` để chứng nhận DB cũ là do code mới tạo; guard phải từ chối nếu fingerprint không khớp.

Ở đầu mỗi lệnh train/evaluate/predict/resume, pipeline nhận chính xác `--db` sẽ mở, tính SHA-256 từ nội dung DB, đối chiếu byte count và SHA kỳ vọng trong `artifact_manifest.json` cạnh DB, đồng thời xác minh SQL schema và các mapping. Fingerprint tương thích dùng định danh nội dung, không dùng đường dẫn cài đặt; đường dẫn thật chỉ lưu trong trace. Không có cache theo kích thước. Lệnh này đọc toàn bộ DB để xác minh; lượt rà mã hiện tại không chạy lệnh đó. Resume dựng model và Adam slots trước strict restore, rồi kiểm tra checkpoint shards, optimizer iteration, update counter, schedule index, số candidate đã tiêu thụ và khóa đầy đủ của candidate cuối cùng trong prefix được replay. Checkpoint cũ thiếu cursor/key mới hoặc chưa được xác minh theo hợp đồng này sẽ bị từ chối thay vì được chứng nhận tương thích.

### Bước 2 — Tạo batch theo candidate và lookup history/news

Code chính: [`mind_data_pipeline/data.py`](mind_data_pipeline/data.py), [`mind_news_metadata/batch_data.py`](mind_news_metadata/batch_data.py).

Loader lấy một candidate cùng impression/user/history tương ứng, tra news lookup theo DID rồi dựng tensor cố định chiều dài. Candidate order và metadata impression được giữ ngoài feature tensors; labels không được đưa vào encoder. News metadata được cache có giới hạn bộ nhớ. UID ngoài phạm vi embedding được ánh xạ sang UNK theo hợp đồng local, còn mapping UID đầy đủ vẫn được giữ để truy vết.

### Bước 3 — Tạo 15 field representations

Code chính: [`mind_news_metadata/field_encoder.py`](mind_news_metadata/field_encoder.py), hợp đồng tại [`mind_news_metadata/field_contract.json`](mind_news_metadata/field_contract.json). Mỗi output có shape `[B,100]`, dtype float32.

| # | Field | Cách tạo và lịch sử/query |
|---:|---|---|
| 1 | `uid` | Lookup user embedding. |
| 2 | `did` | Lookup candidate news embedding. |
| 3 | `his_id` | Attention pooling trên tối đa 200 history DID; query-free là giả định tái triển khai vì semantics của `PoolingWrapper('att')` chưa khôi phục được. |
| 4 | `cat` | Lookup category của candidate. |
| 5 | `sub_cat` | Lookup subcategory của candidate. |
| 6 | `title_entities` | Cộng entity và entity-type embedding theo vị trí rồi attention-pool title entities. |
| 7 | `abstract_entities` | Cùng cách tạo cho abstract entities. |
| 8 | `his_title_merge_entities` | DIN trên tối đa hai entity mỗi news trong tối đa 50 title history; query là `title_entities`. |
| 9 | `his_abstract_merge_entities` | DIN trên tối đa hai entity mỗi news trong tối đa 10 abstract history; query là `abstract_entities`. |
| 10 | `his_cats` | Attention pooling trên category history. |
| 11 | `his_sub_cats` | Cùng attention layer với `his_cats`, áp dụng cho subcategory history. |
| 12 | `cur_title` | Word embedding + attention pooling trên title, tối đa 30 token. |
| 13 | `his_title` | Encode từng title history bằng encoder title dùng chung; DIN query là `cur_title`, tối đa 50 title. |
| 14 | `cur_abstract` | Word embedding + attention pooling trên abstract, tối đa 50 token. |
| 15 | `his_abstract` | Encode từng abstract history bằng encoder abstract dùng chung; DIN query là `cur_abstract`, tối đa 10 abstract. |

Tokenizer là BERT base cased tokenizer; mô hình **không** chạy BERT encoder. GloVe được tạo từ title+abstract metadata train/dev/test, không đọc click labels. Các vector word/entity local là artifacts tái triển khai, không phải file pretrained của tác giả. Entity và type embedding được cộng trước pooling; exact behavior của mask/length trong dependency riêng vẫn chưa xác minh.

Chia sẻ trọng số: một word table dùng cho title và abstract; encoder candidate/history được dùng chung trong mỗi nhánh text; entity và type tables dùng chung giữa title/abstract và history; pool candidate entity dùng chung; pool category/subcategory history dùng chung; DIN title và abstract tách riêng; một entity DIN dùng chung cho hai nhánh entity history. Không có `cur_body`/`his_body` dưới dạng input, parameter hoặc vector placeholder.

Attention/DIN local có masking PAD, zero mass tại vị trí padding và output zero cho history rỗng. Công thức local được mô tả trong source matrix, nhưng không tuyên bố tương đương số học với `mt.layers.Pooling`, `PoolingWrapper` hoặc `DinAttention` riêng của upstream.

### Bước 4 — Feature interaction và prediction

Code chính: [`mind_news_metadata/recommendation_model.py`](mind_news_metadata/recommendation_model.py).

1. Stack 15 field vectors thành `[B,15,100]`.
2. Flatten thành 1500 giá trị.
3. Tính dot product cho từng cặp `i < j`: 105 giá trị.
4. Nối thành vector interaction 1605 chiều.
5. MLP 1024 → 512 → 256 với ReLU; dense scalar tạo logit; sigmoid tạo click probability.

Report mô tả `concat + dot` trước MLP. Thứ tự và nội bộ `melt.Pooling('concat,dot')` không công khai đủ để kết luận local output tương đương số học.

### Bước 5 — Sampling, loss và training

Code chính: [`mind_news_metadata/recommendation_protocol.py`](mind_news_metadata/recommendation_protocol.py), [`mind_news_metadata/full_pipeline.py`](mind_news_metadata/full_pipeline.py).

- Giữ mọi positive; chia negative thành năm partition. Local dùng seed theo impression để tạo partition xác định. Selection rule giống mô tả báo cáo, nhưng chuỗi RNG khác process-global shuffle upstream; danh sách candidate negative có thể khác.
- Shuffle thứ tự impression train bằng seed cấu hình; loader đọc candidate theo impression thay vì ngẫu nhiên hóa toàn bộ candidate.
- Loss là sigmoid cross-entropy từ logits, lấy trung bình trên số candidate được chọn trong effective batch. Test labels null và không được chuyển thành negative.
- Effective batch cấu hình 1024, microbatch hiện đặt 16 và tích lũy gradient trước optimizer update. Đây là execution choice của local pipeline.
- Loader dựng history một lần cho mỗi impression trong batch và broadcast các mảng history đã materialize sang candidate tương ứng. Training giữ metadata candidate tối thiểu để checkpoint và resume xác minh cursor bằng khóa `(split, impression_id, candidate_position, line_no, candidate_id)`; không bỏ metadata này. Training loop gom scalar loss trong một effective batch rồi mới đồng bộ để đọc log. Đây là thay đổi thực thi dự định giữ nguyên model, labels, sampling và effective batch; chưa có runtime parity hoặc benchmark trong lượt rà mã nên chưa có speedup được xác nhận.
- Local dùng Keras Adam, beta1=0.9, beta2=0.999, epsilon=1e-7; weight decay, clipping, EMA và các option khác được ghi rõ trong config. Không coi đây là bản khôi phục `bert-adam`.
- Schedule local: warmup tuyến tính với `ceil(0.1 × số update)`, sau đó giảm tuyến tính đến learning rate tối thiểu. Báo cáo gọi schedule là triangle LR; scheduler implementation và step convention gốc chưa được xác minh đầy đủ.
- Checkpoint chứa model, optimizer, update/schedule counters, số candidate đã tiêu thụ, sampling cursor và candidate key cuối cùng. Cursor được xác minh bằng replay từ đầu đến đúng candidate key, kể cả khi epoch kết thúc bằng effective batch ngắn. Checkpoint update 50 đang lưu là trạng thái từng phần; trạng thái tương thích với schema cursor mới chưa được runtime xác minh. Mọi resume phải qua fingerprint guard; không sửa metadata checkpoint để ép tương thích.

### Bước 6 — Evaluation, prediction và ensemble

Code chính: [`mind_news_metadata/prediction.py`](mind_news_metadata/prediction.py), metrics tại `recommendation_protocol.py` và evaluator trong `full_pipeline.py`.

- Metrics được tính theo impression: AUC, MRR và nDCG@5/@10. MRR local là trung bình reciprocal rank của các positive trong impression. Cách xử lý ties/nhóm rỗng được ghi trong source matrix vì các evaluator upstream có hơn một đường aggregation.
- Report chọn first 1M dev instances và mask 92% candidate DID trong protocol nêu kết quả; cấu hình local có thể đánh giá full dev. Dev vẫn là dữ liệu chọn mô hình, không phải holdout độc lập.
- Test không có nhãn: chỉ tạo prediction, không tính metric bằng nhãn giả.
- Ensemble local ghép prediction theo candidate key rồi trung bình probability. Đây là interface; chưa có ensemble hoàn tất nào được tuyên bố.

## 4. Đối chiếu module local với source thực thi

| Module trong dự án | File/symbol upstream gần nhất | Vai trò và khác biệt chính |
|---|---|---|
| `configs/mind_no_body_15field.json`, `mind_config.py` | `src/train/v17/{body,base,common}.sh`, `src/config.py`, `prepare/config.py` | Gom cấu hình local vào một nguồn; v17 bật body; shell có batch 4096/1024 xung đột chưa phân giải hoàn toàn. |
| `mind_data_pipeline/preprocess.py` | `prepare/gen-dict.py`, `gen-dict-news.py`, `gen-records.py` và notebook trong `juypter/` | Tạo mapping/SQLite thay TFRecords; có policy duplicate; giữ history count quirk; sửa abstract entity stale variable. |
| `mind_data_pipeline/data.py`, `mind_news_metadata/batch_data.py` | `src/dataset.py`, `src/util.py:adjust/get_id`, `src/model.py` | Đọc SQLite và materialize candidate batches; thay loader TFRecord và dependency lookup riêng. |
| `word_embedding.py`, `normalize_word_matrix.py`, `build_entity_matrix.py` | `src/tools/merge-word-emb.py`, `merge-entity-emb.py`, `src/util.py:create_emb` | Tạo/căn artifacts local. Không phải matrix gốc của tác giả. |
| `field_encoder.py` | `src/model.py:Encoder/SeqsEncoder/Encoders/Model`, `src/util.py:get_att_pooling/create_emb` | Tạo 15 field; local attention/DIN thay dependency private, nên công thức nội bộ chưa chứng minh trùng. |
| `recommendation_model.py` | `src/model.py` interaction, MLP, sigmoid | Hiện thực concat/dot và head local; private `melt.Pooling`/MLP chưa được so số học. |
| `recommendation_protocol.py` | `prepare/gen-records.py`, `src/config.py`, `src/eval/evaluate.py`, `src/evaluate.py` | Sampling/LR/metrics local có seed, tie-break và aggregation được công bố; một số semantics khác source. |
| `full_pipeline.py` | `src/train.py`, `src/loss.py`, `melt.fit` ngoài snapshot | Training/checkpoint local; Keras Adam/BCE aggregation không được khẳng định tương đương upstream harness. |
| `prediction.py` | `src/evaluate.py`, `src/eval/evaluate.py`, `src/tools/prob2rank.py`, notebook ensemble | Prediction và keyed probability ensemble local; chưa tuyên bố giống submission pipeline gốc. |

> Bảng bằng chứng chi tiết theo trang, file và symbol nằm ở [`docs/source_reimplementation_matrix.md`](docs/source_reimplementation_matrix.md).

## 5. Bố cục repository

- `mind_data_pipeline/`: parsing raw TSV, vocabulary/mapping, SQLite, fingerprint và loader Stage 01.
- `mind_news_metadata/`: artifact builders, field encoder, interaction/head, sampling, optimizer/schedule, checkpoint, metrics và prediction.
- `configs/`, `mind_config.py`: cấu hình chuẩn và các view cho preprocessing/encoder/training.
- `data/`: raw MIND train/dev/test ở máy cục bộ; pipeline không sửa raw.
- `artifacts/mind_large_prepared/`: DB, vocabulary, tokenizer mapping, GloVe/entity matrices và provenance; không được đưa lên Git.
- `artifacts/recommendation_runs/`: run metadata và checkpoint cục bộ.
- `upstream-mind/`: snapshot khóa chỉ dùng tham chiếu; không chỉnh sửa.

## 6. Môi trường và lệnh chạy cục bộ

`requirements-mind-training.txt` mô tả môi trường tương thích TensorFlow CPU 2.15.1, không phải runtime lịch sử upstream:

```bash
python3.11 -m venv .venv-mind
.venv-mind/bin/python -m pip install -r requirements-mind-training.txt
```

Kiểm tra DB/artifact hiện có theo manifest:

```bash
PYTHONPATH=. .venv-mind/bin/python -m mind_news_metadata.data_validation \
  --data-root data --db artifacts/mind_large_prepared/stage01.sqlite \
  --out artifacts/mind_large_prepared
```

Chỉ chạy lệnh sau khi chủ động muốn tạo lại Stage 01 từ raw bằng output directory phù hợp với fingerprint hiện hành:

```bash
PYTHONPATH=. .venv-mind/bin/python -m mind_data_pipeline.preprocess \
  --data-root data --out artifacts/mind_large_prepared
```

Train partition 0 từ đầu trong thư mục output riêng:

```bash
PYTHONPATH=. .venv-mind/bin/python -m mind_news_metadata.full_pipeline train \
  --partition 0 --effective-batch 1024 --microbatch 16 \
  --output-dir artifacts/recommendation_runs/partition-0
```

Lệnh train trên không được chạy nếu mục tiêu chỉ là đọc tài liệu hoặc kiểm tra cấu hình. `--resume` chỉ dùng khi run config, code fingerprint, input artifacts, partition, seed, batch và checkpoint khớp; không bỏ qua guard. Training lưu tối đa hai checkpoint gần nhất theo cấu hình, hiện mỗi 25 optimizer updates.

## 7. Trạng thái và giới hạn kết luận

- **Đã triển khai pipeline và rà soát tĩnh; chưa nghiệm thu runtime trước huấn luyện.** Đây là biến thể tái triển khai 15 field không body, không phải implementation nguyên bản hoặc tương đương số học với tác giả.
- chưa kiểm tra undefined names bằng linter. Môi trường hiện không có `ruff`, `pyflakes` hoặc `flake8`.
- Membership artifact mới chưa được tạo. DB và checkpoint chưa được xác minh với mã hiện hành; model runtime, strict resume và notebook Kaggle hiện tại chưa chạy.
- Checkpoint update 50 được bảo toàn, chưa xác nhận tương thích. Không sửa fingerprint hoặc bỏ strict guard để ép resume.
- GloVe/entity ban đầu không đồng nghĩa attention/DIN/embedding của recommender đã học từ click labels. Chưa có kết quả training được nghiệm thu.

### A. Chuẩn bị và xác minh trước training

Membership cần cho `dev_new_did_masked` phải được tạo từ raw train behaviors: DID trong history và candidates, không dùng nhãn hay news metadata. Lệnh CLI hiện có:

```bash
PYTHONPATH=. .venv-mind/bin/python -m mind_news_metadata.full_pipeline build-membership \
  --db artifacts/mind_large_prepared/stage01.sqlite \
  --train-behaviors data/MINDlarge_train/behaviors.tsv \
  --output artifacts/recommendation_runs/train_did_membership.npy
```

Sau đó có thể kiểm tra DB và model-only restore/forward trên một candidate dev bằng CLI prediction hiện có; output smoke được ghi dưới `/tmp`, không tính metric:

```bash
PYTHONPATH=. .venv-mind/bin/python -m mind_news_metadata.full_pipeline predict \
  --run-dir artifacts/recommendation_runs/partition-0-cpu-full \
  --db artifacts/mind_large_prepared/stage01.sqlite \
  --split dev --profile dev_raw --max-candidates 1 \
  --output /tmp/mind-15field-dev-smoke.csv.gz
```

CLI hiện không có chế độ strict-restore-only. Lệnh `train --resume` dưới đây là lệnh tiếp tục training: nếu restore thành công, optimizer sẽ cập nhật tiếp; không dùng như một probe không ghi trạng thái.

```bash
PYTHONPATH=. .venv-mind/bin/python -m mind_news_metadata.full_pipeline train \
  --db artifacts/mind_large_prepared/stage01.sqlite \
  --partition 0 --seed 20261110 --effective-batch 1024 --microbatch 16 \
  --resume --output-dir artifacts/recommendation_runs/partition-0-cpu-full
```

Chỉ chạy resume sau khi kiểm tra DB/model, xác nhận run config và quyết định tiếp tục training. Guard có thể từ chối checkpoint cũ; không được bỏ qua lỗi đó.

### B. Khác biệt nguồn chưa khôi phục được

Các điểm này vẫn **UNVERIFIED**, không được xem là đã giải quyết chỉ vì đã ghi lại: liên kết implementation `bert-adam`; công thức và step convention scheduler; phép toán/masking chính xác của pooling và DIN riêng; RNG lịch sử; provenance tokenizer/GloVe so với artifact tác giả. Profile dev hiện theo ý định báo cáo và khác hành vi executable upstream đã khóa. Chi tiết nằm trong [`docs/source_reimplementation_matrix.md`](docs/source_reimplementation_matrix.md).

- Không tuyên bố tái hiện nguyên bản, metric cuộc thi, chất lượng tương đương body model hoặc ensemble tác giả.
