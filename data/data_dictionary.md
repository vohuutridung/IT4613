# Data Dictionary — MIND-large

Tài liệu mô tả schema của các file MIND-large trong `MINDlarge_train/`, `MINDlarge_dev/` và `MINDlarge_test/`. Các file TSV không có dòng tiêu đề; tên cột dưới đây theo thứ tự trường trong từng dòng. Schema được đối chiếu với tài liệu [Microsoft Learn về MIND](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news).

## 1. `behaviors.tsv`

Mỗi dòng là một impression (lần hiển thị danh sách tin cho một người dùng). File có 5 trường, phân cách bằng tab.

| # | Tên cột | Kiểu dữ liệu gợi ý | Ý nghĩa / định dạng |
|---:|---|---|---|
| 1 | `impression_id` | số nguyên hoặc chuỗi | Mã impression trong split. Nên giữ dạng chuỗi nếu muốn bảo toàn nguyên trạng. |
| 2 | `user_id` | chuỗi | Mã người dùng đã ẩn danh. |
| 3 | `time` | chuỗi ngày giờ → datetime | Thời điểm impression, định dạng `MM/DD/YYYY hh:mm:ss AM/PM`; dữ liệu không ghi rõ múi giờ. |
| 4 | `history` | danh sách chuỗi | Các mã tin người dùng đã nhấp trước impression này, phân cách bằng dấu cách. Có thể rỗng. |
| 5 | `impressions` | danh sách candidate | Train/dev: tin được hiển thị, phân cách bằng dấu cách; mỗi phần tử có dạng `news_id-click_label` (`1` click, `0` không click). Test: chỉ có `news_id`, không có nhãn click. |

## 2. `news.tsv`

Mỗi dòng là một bài báo. File có 8 trường, phân cách bằng tab.

| # | Tên cột | Kiểu dữ liệu gợi ý | Ý nghĩa / định dạng |
|---:|---|---|---|
| 1 | `news_id` | chuỗi | Mã bài báo, ví dụ `N55528`. Là khóa để nối với `history` và `impressions`. |
| 2 | `category` | chuỗi / categorical | Danh mục chính của bài báo, ví dụ `sports`, `news`, `finance`. |
| 3 | `subcategory` | chuỗi / categorical | Danh mục con, ví dụ `lifestyleroyals`. |
| 4 | `title` | chuỗi | Tiêu đề bài báo. |
| 5 | `abstract` | chuỗi nullable | Tóm tắt bài báo; có thể thiếu hoặc rỗng. |
| 6 | `url` | chuỗi | URL của bài viết nguồn. |
| 7 | `title_entities` | JSON array dạng chuỗi | Thực thể tri thức được liên kết trong tiêu đề. Phần tử có các thuộc tính như `Label`, `Type`, `WikidataId`, `Confidence`, `OccurrenceOffsets`, `SurfaceForms`; có thể là mảng rỗng. |
| 8 | `abstract_entities` | JSON array dạng chuỗi | Thực thể tri thức được liên kết trong phần tóm tắt; cấu trúc giống `title_entities`, có thể là mảng rỗng. |

Các trường entity là JSON được lưu bên trong một trường TSV, vì vậy cần parse JSON sau khi đọc TSV. `WikidataId` là mã thực thể; `OccurrenceOffsets` và `SurfaceForms` mô tả vị trí/chuỗi văn bản tương ứng trong nội dung.

## 3. `entity_embedding.vec`

Mỗi dòng gồm `entity_id` và vector embedding của thực thể, phân cách bằng tab. Dữ liệu đã kiểm tra có 100 giá trị số cho mỗi vector. File vật lý có thêm một trường rỗng ở cuối dòng, do tab kết thúc dòng; khi parse cần bỏ trường rỗng cuối nếu có.

| Trường | Kiểu dữ liệu gợi ý | Ý nghĩa |
|---|---|---|
| `entity_id` | chuỗi | Mã thực thể Wikidata, ví dụ `Q...`. |
| `embedding[1..100]` | 100 số thực | Vector 100 chiều học được bằng TransE. |

## 4. `relation_embedding.vec`

Tương tự file entity embedding, mỗi dòng gồm mã quan hệ và vector 100 chiều. Dữ liệu thực tế có 100 số thực cùng một trường rỗng kết thúc dòng.

| Trường | Kiểu dữ liệu gợi ý | Ý nghĩa |
|---|---|---|
| `relation_id` | chuỗi | Mã quan hệ trong knowledge graph, ví dụ `P...`. |
| `embedding[1..100]` | 100 số thực | Vector biểu diễn quan hệ. |

## Quan hệ giữa các file

- `behaviors.history[*]` và `behaviors.impressions[*].news_id` tham chiếu `news.news_id` trong cùng split.
- `news.title_entities[*].WikidataId` và `news.abstract_entities[*].WikidataId` có thể tra trong `entity_embedding.vec`; không phải thực thể nào cũng có vector.
- `relation_embedding.vec` chứa vector cho các quan hệ của knowledge graph; nó không nối trực tiếp với các cột hành vi.
- `MINDlarge_train`, `MINDlarge_dev` và `MINDlarge_test` là các split riêng. Khi phân tích, cần giữ rõ nguồn split để tránh gộp nhầm dữ liệu.

## Ghi chú khi đọc dữ liệu

- Vì TSV không có header, cần tự gán tên cột theo thứ tự ở trên.
- Tách trường bằng tab, không coi dấu `"` là dấu bao trường CSV. Một số giá trị văn bản có dấu ngoặc kép không cân bằng; parser CSV mặc định có thể gộp nhầm cột.
- Đọc `news_id`, `user_id` và entity/relation ID dưới dạng chuỗi; không chuyển ID sang số.
- Tách `history` theo khoảng trắng; tách mỗi token trong `impressions` tại dấu `-` cuối để lấy nhãn.
- Trong test, candidate không có hậu tố nhãn nên không thể tính CTR hoặc metric click trực tiếp.
- Kiểm tra chuỗi rỗng trước khi parse JSON hoặc chuyển đổi kiểu.
- Tài liệu Microsoft mô tả các embedding là 100 chiều; việc kiểm tra file trong workspace xác nhận 100 số thực và một tab rỗng ở cuối mỗi dòng.
