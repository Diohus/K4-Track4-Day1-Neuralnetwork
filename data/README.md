# Dữ liệu: Forest CoverType

Nguồn: Blackard & Dean (1999), UCI Machine Learning Repository, *Covertype*. Dự đoán loại thảm thực vật rừng (7 lớp) cho các ô đất 30 m × 30 m ở Roosevelt National Forest, Colorado, từ biến địa hình.

## Các file

| File | Sinh viên làm gì | Nội dung |
|---|---|---|
| `covtype.csv.gz` | chỉ đọc | 581 012 dòng, 55 cột: 54 đặc trưng + `Cover_Type` (1..7). Có tiêu đề. |
| `split_metadata.csv` | chỉ đọc, **không sửa** | 581 012 dòng, 2 cột: `row_id` (0..581 011, là chỉ số dòng trong `covtype.csv.gz`), `split` ∈ {`train`, `eval`} |
| `processed/train.npz`, `processed/eval.npz` | **sinh ra** bởi `python scripts/split_data.py` (đã nằm trong `.gitignore`, không đưa lên git) | `X` (float32, N×54), `y` (int64, 0..6), `row_id` (int64), `feature_names` |

## Phép chia

- Phân tầng theo nhãn, seed 42: `train` = 464 809 mẫu (80%), `eval` = 116 203 mẫu (20%).
- Chỉ có hai tập. **Không có validation**: bạn tự tách validation từ `train` (xem GUIDE, Part 0).
- `eval` chỉ dùng để chấm điểm cuối (xem GUIDE, mục *Cách đánh giá*).
- Tạo lại metadata (chỉ giảng viên cần): `python scripts/make_split_metadata.py`.

## Đặc trưng (54 cột)

| Cột | Số lượng | Loại |
|---|---|---|
| `Elevation`, `Aspect`, `Slope`, `Horizontal_Distance_To_Hydrology`, `Vertical_Distance_To_Hydrology`, `Horizontal_Distance_To_Roadways`, `Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm`, `Horizontal_Distance_To_Fire_Points` | 10 (cột 0..9) | số liên tục (số nguyên), **cần chuẩn hoá** |
| `Wilderness_Area_0..3` | 4 | one-hot nhị phân |
| `Soil_Type_0..39` | 40 | one-hot nhị phân |

## Nhãn

Trong file CSV gốc `Cover_Type` là 1..7; sau `split_data.py`, `y = Cover_Type − 1` (0..6).

| `y` | Loại rừng | Tỉ lệ (≈) |
|---|---|---|
| 0 | Spruce/Fir | 36,5% |
| 1 | Lodgepole Pine | 48,8% |
| 2 | Ponderosa Pine | 6,2% |
| 3 | Cottonwood/Willow | 0,5% |
| 4 | Aspen | 1,6% |
| 5 | Douglas-fir | 3,0% |

| 6 | Krummholz | 3,5% |

Dữ liệu **mất cân bằng**: đoán luôn lớp 1 cho accuracy 0,4876 nhưng macro-F1 chỉ ≈ 0,094. Vì vậy chỉ số chính của lab là macro-F1.

## Lưu ý về phép chia gốc của dataset

Bài báo gốc dùng phép chia khác: 11 340 mẫu đầu (cân bằng 1 620 mẫu/lớp) làm train, 3 780 mẫu kế làm validation, 565 892 mẫu còn lại làm test. Lab này **không dùng phép chia đó**, nên không so sánh điểm của bạn với con số 70,58% trong bài báo gốc (đo trên chỉ 11 340 mẫu train).
