# Chạy lại bài lab

Thư mục nộp chứa toàn bộ code và notebook. Dữ liệu, script chấm và mẫu Excel nằm trong repo gốc, bên cạnh thư mục nộp.

## Môi trường

Python 3.10+; cài thư viện với `python -m pip install -r code/requirements.txt` khi đứng trong thư mục nộp. Colab/Kaggle thường có sẵn các thư viện khoa học; `openpyxl` cần cho Excel.

Windows GTX 1650 trong lần chạy gốc dùng Python 3.13, PyTorch 2.7.1+cu126. Lệnh cài GPU đã kiểm tra từ tài liệu chính thức: `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126`. Tham khảo https://pytorch.org/get-started/previous-versions/ nếu dùng môi trường khác. Các phiên bản đã đo được ghi ở `environment_versions.json` và `run_manifest.json`.

## Notebook

Mở `lab.ipynb` từ thư mục `code/`, chọn **Restart & Run All**. Nếu notebook server bắt đầu ở repo gốc, ô đầu tự tìm `submission_*/code/`. Đường dẫn chuẩn từ thư mục nộp/code là `../../data` và `../../scripts`. Với dữ liệu đặt ở nơi khác, đặt biến môi trường `LAB_REPO_ROOT` tới repo chứa `data/`, `scripts/`, `templates/` trước khi tạo `Lab`.

Trên Colab/Kaggle: tải và giải nén bộ nộp; tải/clone repo dữ liệu gốc, đặt `CODE_DIR` trong ô đầu đến thư mục nộp/code và `os.environ['LAB_REPO_ROOT']` đến repo dữ liệu, bật GPU rồi chạy tất cả. Không chỉ tải riêng một notebook vì notebook import các module của bộ nộp.

`REUSE_RESULTS=True` dùng lịch sử đo đã lưu để kiểm tra và tạo lại ảnh/bảng/báo cáo; không giả định đây là một lần huấn luyện mới. `REUSE_RESULTS=False` chạy lại mọi thí nghiệm trong một thư mục mới, để không dùng điểm eval cũ khi chọn cấu hình. Cùng seed vẫn có thể khác đôi chút trên phần cứng hoặc phiên bản PyTorch khác.

## Dòng lệnh

Từ repo gốc:

```text
python submission_MSSV/code/workflow.py --out submission_MSSV --stage all
python submission_MSSV/code/validate_submission.py submission_MSSV
```

Các stage `health`, `search`, `topics`, `final` có thể chạy riêng. Dò lr và mỗi thí nghiệm chạy 20 epoch. Chọn checkpoint bằng val loss; chọn cấu hình bằng val macro-F1 ở checkpoint đó. Chỉ baseline và cấu hình cuối được đánh giá bằng script chính thức.

Để tạo một nghiên cứu mới, dùng thư mục đầu ra khác, ví dụ `--out submission_MSSV_rerun --stage all`. Không điều chỉnh cấu hình của nghiên cứu đã chấm eval. Mỗi run lưu JSON/ảnh ngay sau khi chạy; các checkpoint dùng để tiếp tục nằm ở `.work/<tên thư mục nộp>/` của repo, không đóng gói vào bài nộp.

## Điền thông tin sinh viên

Sửa `profile.json` bằng họ tên/MSSV thật, đổi tên thư mục và zip thành `submission_<MSSV>`, rồi chạy lại notebook để tạo báo cáo với thông tin mới. Những số liệu thí nghiệm không phụ thuộc họ tên/MSSV.
