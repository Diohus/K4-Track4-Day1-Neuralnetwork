"""Build the Part 0-4 notebook. Outputs are filled by execution, never fabricated."""
import json
from pathlib import Path
import nbformat as nbf
from experiment_plan import HYPOTHESES

def build(out):
    out=Path(out); cells=[]
    md=lambda s:cells.append(nbf.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbf.v4.new_code_cell(s))
    md("# Lab Day 1 — MLP và thí nghiệm huấn luyện\n\n"
       "Notebook Part 0–4. Chạy cùng các module trong thư mục `code/`. "
       "Trên Colab/Kaggle, tải toàn bộ thư mục nộp; giữ thư mục dữ liệu repo bên cạnh thư mục nộp, "
       "hoặc đặt `LAB_REPO_ROOT` đến repo có `data/`, `scripts/`, `templates/`. "
       "Bật GPU. Cài `pip install -r code/requirements.txt` nếu môi trường thiếu thư viện.\n\n"
       "`REUSE_RESULTS=True` đọc lịch sử đo đã lưu và tái tạo bảng/ảnh/báo cáo; "
       "không coi việc đọc lịch sử là huấn luyện mới. Để huấn luyện lại, đặt False: kết quả "
       "ghi trong thư mục mới, không chỉnh lựa chọn của nghiên cứu đã chấm eval.")
    code('''from pathlib import Path
import os, sys, json, numpy as np, pandas as pd, torch
from IPython.display import display, Image, Markdown

# If Jupyter starts at the repository root, locate the submitted modules.
CODE_DIR = Path.cwd()
if not (CODE_DIR / "workflow.py").exists():
    matches = sorted(CODE_DIR.glob("submission_*/code/workflow.py"))
    CODE_DIR = matches[0].parent if matches else CODE_DIR / "code"
sys.path.insert(0, str(CODE_DIR.resolve()))
from workflow import Lab
from checks import validate_numerics
from results_table import load_results
REUSE_RESULTS = True
OUT_DIR = CODE_DIR.resolve().parent
if not OUT_DIR.name.startswith("submission_"):
    OUT_DIR = OUT_DIR / "submission_MSSV"
if not REUSE_RESULTS:
    from datetime import datetime
    OUT_DIR = OUT_DIR.parent / (OUT_DIR.name + "_rerun_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
lab = Lab(out=OUT_DIR)
print("PyTorch", torch.__version__, "device", lab.device)
print("Outputs:", OUT_DIR)
validate_numerics()''')
    md("## Part 0 — Dữ liệu\n\nTrain/eval theo metadata cố định; validation tách từ train, stratified seed 42. "
       "Chỉ chuẩn hóa 10 cột số bằng train sau tách. Giữ mọi dòng ở batch cuối.")
    code('''for name in ("tr", "val", "eval"):
    print(name, tuple(lab.data[f"X_{name}"].shape), lab.data[f"X_{name}"].dtype, lab.data[f"y_{name}"].dtype)
print("Numeric mean:", lab.data["X_tr"][:, :10].mean(0).cpu().numpy())
print("Numeric std:", lab.data["X_tr"][:, :10].std(0, correction=0).cpu().numpy())
print("Majority val accuracy:", lab.data["majority_val_acc"])''')
    md("## Part 1 — Model và sức khỏe ban đầu\n\nDự đoán: logits gần đều cho CE gần ln7; "
       "He lớp cuối có thể làm logits không đều. Model đúng phải học thuộc được 20 mẫu và có gradient ở mọi tham số.")
    code('''health = lab.health()
display(pd.DataFrame(health["shapes"], columns=["layer", "shape"]))
print("Parameters:", health["parameters"], "step0:", health["step0_loss"], "ln7:", np.log(7))
print("Gradient norms:", health["grad_norms"])
print("Overfit loss/acc:", health["overfit_loss"], health["overfit_acc"])
display(Image(filename=str(OUT_DIR / "figures/health-overfit20.png")))
display(pd.DataFrame(health["initialization"]).T)''')
    md("**Nhận xét:** kiểm tra shape/gradient và batch nhỏ là bằng chứng pipeline hoạt động. "
       "Không ép loss bước 0 bằng ln7: đây là mốc cho logits đều, không phải định luật của mọi khởi tạo He.")
    md("## Part 2 — Chọn lr và baseline\n\n"+HYPOTHESES["optimizer"]+
       " Dò SGD momentum: 0.01/0.05/0.1; Adam: 0.0003/0.001/0.003, cùng seed 1 và 20 epoch. "
       "Sau khi chọn lr baseline bằng val, chạy seed 1/2/3 để đo dao động.")
    code('''search_results = lab.search()
display(pd.DataFrame([dict(exp_id=r["cfg"]["exp_id"], optimizer=r["cfg"]["optimizer"], lr=r["cfg"]["lr"],
    best_epoch=r["summary"]["best_epoch"], val_f1=r["summary"]["val_macro_f1"]) for r in search_results]))
baselines = [lab.run(lab.baseline_cfg(s)) for s in (1, 2, 3)]
base_f1 = np.array([r["summary"]["val_macro_f1"] for r in baselines])
noise = 2 * base_f1.std(ddof=1)
print("Baseline F1 mean ± sample std:", base_f1.mean(), base_f1.std(ddof=1), "2 sigma:", noise)
display(Image(filename=str(OUT_DIR / "figures/base-s1.png")))''')
    md("## Part 3 — Thí nghiệm\n\nMỗi cấu hình ghi giả thuyết trước khi chạy; giữ split/seed/budget. "
       "Cặp lr cao chỉ khác clipping trong cặp. Ngưỡng clipping lấy từ baseline. "
       "FP16 dùng GradScaler và unscale trước đo/clip; BF16 chỉ chạy nếu có phần cứng native.")
    md("### Giả thuyết trước các thí nghiệm\n\n"+"\n\n".join(f"**{g}:** {h}" for g,h in HYPOTHESES.items() if g not in ("baseline","optimizer")))
    code('''topic_results = lab.topics()
all_results = load_results(OUT_DIR / "results")
display(pd.DataFrame([dict(exp_id=r["cfg"]["exp_id"], group=r["cfg"]["group"],
    f1=r["summary"]["val_macro_f1"], delta_vs_base_mean=r["summary"]["val_macro_f1"]-base_f1.mean(),
    beyond_2sigma=abs(r["summary"]["val_macro_f1"]-base_f1.mean())>noise,
    time_s=r["summary"]["time_per_epoch_s"], diverged=r["summary"]["diverged"]) for r in all_results]))
for group in ("loss", "optimizer", "hparam", "dropout", "clipping", "amp", "init"):
    figure = OUT_DIR / "figures" / f"compare_{group}.png"
    if figure.exists(): display(Image(filename=str(figure)))''')
    # Add measured post-run observations if results exist, with IDs and noise caveat.
    results_dir=out/"results"
    if results_dir.exists():
        observed=[]
        for p in sorted(results_dir.glob("*.json")):
            if p.name=="health.json": continue
            r=json.loads(p.read_text(encoding="utf-8"))
            observed.append(f"`{r['cfg']['exp_id']}`: val macro-F1 {r['summary']['val_macro_f1']:.6f}, "
                            f"best epoch {r['summary']['best_epoch']}, diverged {r['summary']['diverged']}.")
        md("### Đối chiếu sau chạy\n\n"+"\n\n".join(observed)+
           "\n\nCác giá trị trên là lịch sử của bộ nộp hiện tại; nếu huấn luyện lại, dùng bảng output mới. "
           "Giải thích cơ chế, điều khác dự đoán và giới hạn một seed mỗi kỹ thuật nằm ở REPORT.md.")
    md("## Part 4 — Khóa lựa chọn, eval và bộ nộp\n\nChọn cấu hình bằng val F1 tại checkpoint có val loss thấp nhất; "
       "khóa exp_id trước evaluate.py. Chấm baseline và cấu hình cuối. Không điều chỉnh sau khi thấy eval.")
    code('''eval_scores = lab.finalize()
print("Fixed selected experiment:", lab.manifest["selected_exp_id"])
print("Official accuracy/macro-F1:", eval_scores["accuracy"], eval_scores["macro_f1"])
display(pd.DataFrame(eval_scores["per_class"]))
display(Image(filename=str(OUT_DIR / "figures/eval-confusion.png")))
display(Markdown((OUT_DIR / "REPORT.md").read_text(encoding="utf-8")))''')
    code('''from validate_submission import validate_submission
validate_submission(OUT_DIR, lab.repo)
print("Ready:", OUT_DIR / "experiments.xlsx", OUT_DIR / "predictions_eval.csv")''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.13"}})
    nbf.write(nb,out/"code/lab.ipynb")
    return out/"code/lab.ipynb"

if __name__=="__main__":
    import sys
    build(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent.parent)
