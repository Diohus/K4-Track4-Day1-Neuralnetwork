"""Generate a concise report using measured results and official scores only."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from results_table import load_results
from experiment_plan import HYPOTHESES

def write_report(out,manifest):
    out=Path(out)
    results=load_results(out/"results"); by_id={r["cfg"]["exp_id"]:r for r in results}
    bases=[r for r in results if r["cfg"]["group"]=="baseline"]
    base=by_id["base-s1"]; final=by_id[manifest["selected_exp_id"]]
    val_mean=float(np.mean([r["summary"]["val_macro_f1"] for r in bases]))
    sigma=float(np.std([r["summary"]["val_macro_f1"] for r in bases],ddof=1))
    health=json.loads((out/"results/health.json").read_text(encoding="utf-8"))
    ev=json.loads((out/"eval_result.json").read_text())
    be=json.loads((out/"baseline_eval_result.json").read_text())
    base_eval=np.array([be['macro_f1']]+[json.loads((out/f'eval_base_s{s}.json').read_text())['macro_f1'] for s in (2,3)])
    final_eval=np.array([ev['macro_f1']]+[json.loads((out/f'eval_final_s{s}.json').read_text())['macro_f1'] for s in (2,3)])
    paired=final_eval-base_eval
    profile_path=Path(__file__).parent/"profile.json"
    profile=json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    env=manifest["environment"]
    cfg=base["cfg"]
    lines=[f"# Báo cáo Lab Day 1 — {profile.get('name','Chưa cung cấp họ tên')} — {profile.get('student_id','MSSV')}",
        "", "## 1. Thiết lập",
        f"Forest CoverType: train/eval cố định 464.809/116.203; tách val 20%, stratified seed 42 → train 371.847, val 92.962. "
        "Chuẩn hóa 10 cột số bằng mean/std của train sau tách; 44 cột nhị phân giữ nguyên. Eval chỉ dùng sau khi khóa lựa chọn bằng val.",
        f"Môi trường: {env['platform']}; Python {env['python'].split()[0]}, PyTorch {env['torch']}, GPU {env['gpu'] or 'CPU'}. "
        f"M-base 54→256→128→7, ReLU, 47.879 tham số. Baseline: CE, SGD momentum 0,9, lr={cfg['lr']}, "
        "He cho mọi Linear, bias=0, batch=512, 20 epoch, không dropout/clip, FP32, weight decay=0. "
        f"Đoán đa số trên val: accuracy={manifest['majority_val_acc']:.6f} (run_manifest.json; notes baseline trong bảng).",
        "Train loss đo ở eval mode trên 50.000 dòng cố định; val đo toàn bộ. Giữ batch cuối nhỏ hơn 512. "
        "Best checkpoint theo val loss nhỏ nhất; bảng ghi acc/F1 ở checkpoint này. MSE là trung bình trên N×7 phần tử của logits thô so one-hot.",
        "", "## 2. Kiểm tra ban đầu và độ nhiễu",
        f"Shape logits (8,7), assert {health['parameters']} tham số; mọi gradient khác 0 (chi tiết trong notebook). "
        f"Loss CE bước 0={health['step0_loss']:.6f}, so với ln7=1,945910. "
        "ln7 chỉ đúng cho dự đoán đều; He ở lớp cuối có thể tạo logits không đều nên không ép loss về mốc này. "
        "Kiểm tra chuẩn hóa, gradient và học thuộc batch nhỏ dùng để loại trừ lỗi pipeline.",
        f"Học thuộc 20 mẫu cân bằng qua 600 bước Adam lr=0,01: loss={health['overfit_loss']:.8f}, accuracy={health['overfit_acc']:.4f}; "
        "[đường cong](figures/health-overfit20.png). Đây là phép kiểm tra code, không chọn cấu hình bằng 20 mẫu.",
        f"Baseline 3 seed (`base-s1`, `base-s2`, `base-s3`): val macro-F1={val_mean:.6f} ± {sigma:.6f}; "
        f"ngưỡng tham khảo 2σ={2*sigma:.6f}. Val accuracy={np.mean([r['summary']['val_acc'] for r in bases]):.6f} ± "
        f"{np.std([r['summary']['val_acc'] for r in bases],ddof=1):.6f}. "
        "Đây là thước đo dao động baseline với 3 seed, chưa phải kiểm định thống kê hay độ nhiễu riêng của mỗi kỹ thuật.",
        f"Đường baseline: train loss cuối={base['summary']['final_train_loss']:.6f}, "
        f"val loss cuối={base['summary']['final_val_loss']:.6f}, best epoch={base['summary']['best_epoch']}/20. "
        + ("Epoch tốt nhất gần cuối, gợi ý mô hình còn có thể học thêm; chưa có dấu hiệu quá khớp kéo dài chỉ từ đường này. "
           if base['summary']['best_epoch']>=18 else "Epoch tốt nhất xuất hiện trước cuối; quan sát các dao động val trước khi kết luận quá khớp. ")
        + "[Ảnh baseline](figures/base-s1.png).",
        "", "## 3. Kết quả theo chủ đề",
        "Mọi dự đoán được ghi trong experiment_plan.py và run_manifest.json trước tối ưu. Mọi phép so sánh dưới đây dựa trên val; "
        "mỗi exp_id có dòng Excel và ảnh riêng. Các run dò lr cũng chạy 20 epoch.",
        "", "| Chủ đề | exp_id đại diện | best epoch | val macro-F1 | Δ so base-s1 |",
        "|---|---|---:|---:|---:|"]
    representatives={}
    for group in ("loss","optimizer","hparam","dropout","clipping","amp","init"):
        subset=[r for r in results if r["cfg"]["group"]==group]
        if not subset: continue
        representative=max((r for r in subset if not r["summary"]["diverged"]),
                           key=lambda r:r["summary"]["val_macro_f1"],default=subset[0])
        representatives[group]=representative
        s=representative["summary"]; delta=s["val_macro_f1"]-base["summary"]["val_macro_f1"]
        lines.append(f"| {group} | `{representative['cfg']['exp_id']}` | {s['best_epoch']} | {s['val_macro_f1']:.6f} | {delta:+.6f} |")
    def evidence(group):
        r=representatives[group]; delta=r["summary"]["val_macro_f1"]-base["summary"]["val_macro_f1"]
        return (f"Chênh lệch đại diện {delta:+.6f} " + ("vượt" if abs(delta)>2*sigma else "không vượt")+
                f" 2σ baseline; [ảnh so sánh](figures/compare_{group}.png).")
    for group in representatives:
        lines += ["",f"### {group}",f"**Dự đoán:** {HYPOTHESES[group]}"]
        if group=="loss":
            r=by_id["loss-mse"]
            lines.append(f"**Đối chiếu:** MSE đạt F1={r['summary']['val_macro_f1']:.6f}; "
                "CE và MSE không có cùng đơn vị loss. CE tác động qua xác suất lớp, còn MSE logits phạt khoảng cách tới one-hot "
                "và phụ thuộc thang logits; cố định lr CE ở MSE chỉ kiểm tra thay loss tại cấu hình này, chưa tối ưu lr riêng cho MSE.")
        elif group=="optimizer":
            winners=manifest["lr_search_winners"]
            lines += ["**Đối chiếu:** mỗi optimizer được thử 3 lr; so lựa chọn tốt nhất bằng val macro-F1:",
                "", "| Optimizer | exp_id | lr | val F1 |", "|---|---|---:|---:|"]
            for opt,eid in winners.items():
                r=by_id[eid]; lines.append(f"| {opt} | `{eid}` | {r['cfg']['lr']} | {r['summary']['val_macro_f1']:.6f} |")
            lines.append("Adam dùng moment bậc một và hai (betas 0,9/0,999; eps 1e-8), thay đổi bước theo từng tham số; "
                "momentum tích lũy hướng gradient. Khác biệt quan sát phụ thuộc cả optimizer và lr đã dò. "
                "Không thử AdamW nên không kết luận thực nghiệm về Adam so AdamW.")
        elif group=="hparam":
            r=by_id["wide"]
            lines.append(f"**Đối chiếu:** M-wide 161.287 tham số đạt F1={r['summary']['val_macro_f1']:.6f}; "
                f"{r['summary']['time_per_epoch_s']:.2f}s/epoch so baseline {base['summary']['time_per_epoch_s']:.2f}s. "
                "Cùng batch, số epoch và số bước cập nhật; năng lực biểu diễn lớn hơn cũng tăng chi phí tính toán. "
                "20 epoch không bảo đảm các kiến trúc đều hội tụ.")
            if r['summary']['time_per_epoch_s']<base['summary']['time_per_epoch_s']:
                lines.append("Thời gian M-wide trong lần đo này lại thấp hơn baseline, khác dự đoán về wall time. "
                    "Số phép toán lớn hơn không quyết định trực tiếp thời gian của mạng nhỏ trên GPU; "
                    "chi phí kernel và điều kiện máy là giả thuyết, chưa kiểm chứng bằng benchmark lặp.")
        elif group=="dropout":
            r=by_id["drop-0.3"]
            gap=lambda q:q["summary"]["final_val_loss"]-q["summary"]["final_train_loss"]
            lines.append(f"**Đối chiếu:** gap cuối baseline={gap(base):.6f}, dropout 0,3={gap(r):.6f}; "
                f"F1 dropout={r['summary']['val_macro_f1']:.6f}. Dropout làm nhiễu biểu diễn khi train, "
                "giúp chống quá khớp nhưng cũng có thể làm chậm học. Gap nhỏ riêng lẻ chưa đủ để kết luận tổng quát hóa tốt hơn.")
        elif group=="clipping":
            normal=by_id["clip-normal"]
            frac=float(np.mean(normal["history"]["clip_fraction"]))
            lines.append(f"**Đối chiếu:** c={manifest['clip_norm']:.6f}, bằng nửa trung vị mean grad_norm baseline; "
                f"tỉ lệ bước bị clip ở lr thường={frac:.4f}. `highlr-no-clip` và `highlr-clip` cùng lr={cfg['lr']*10:g}, "
                "chỉ khác clipping; đo grad trước clip.")
            for eid in ("highlr-no-clip","highlr-clip"):
                r=by_id[eid]; lines.append(f"`{eid}`: F1={r['summary']['val_macro_f1']:.6f}, "
                    f"completed={r['summary']['completed_epochs']}, diverged={r['summary']['diverged']}.")
            lines.append("Clipping giới hạn chuẩn L2 toàn cục, không đổi hướng; lr cao vẫn có thể gây dao động vì bước cập nhật "
                "và momentum. Việc không phát sinh NaN không đồng nghĩa đã hội tụ tốt.")
            highclip=by_id['highlr-clip']; highplain=by_id['highlr-no-clip']
            lines.append(f"Trong cặp lr cao, clip kích hoạt ở trung bình {np.mean(highclip['history']['clip_fraction']):.4f} "
                f"số bước, chênh F1={highclip['summary']['val_macro_f1']-highplain['summary']['val_macro_f1']:+.6f}. "
                "Không run nào NaN: phép thử không chứng minh clipping cứu phân kỳ; chỉ cho thấy thay đổi quỹ đạo học. "
                "Ngược lại, clip lr thường kích hoạt gần mọi bước, có thể giới hạn bước quá mạnh và làm chậm học.")
        elif group=="amp":
            r=by_id["amp-fp16"]; s=r["summary"]; b=base["summary"]
            lines.append(f"**Đối chiếu:** FP16 {s['time_per_epoch_s']:.2f}s/epoch, peak {s['peak_mem_MB']:.2f} MB; "
                f"FP32 {b['time_per_epoch_s']:.2f}s, peak {b['peak_mem_MB']:.2f} MB. F1 FP16={s['val_macro_f1']:.6f}. "
                + ("FP16 nhanh hơn trong lần đo này. " if s['time_per_epoch_s']<b['time_per_epoch_s'] else "FP16 không nhanh hơn trong lần đo này. ")
                + "Mạng nhỏ chịu chi phí kernel/autocast; không suy rộng sang mạng lớn. FP16 cần loss scaling để hạn chế "
                "underflow; unscale trước khi đo/clip. BF16 có khoảng số rộng hơn và thường không cần scaler, nhưng GPU "
                "này không có BF16 native nên không chạy. Peak bao gồm tensor dữ liệu cùng nằm trên GPU.")
        elif group=="init":
            lines += ["**Đối chiếu:** std sau mỗi hidden ReLU và logits cuối tại bước 0:",
                "", "| Init / exp_id | std theo lớp | CE bước 0 |", "|---|---|---:|"]
            for init,eid in (("he","base-s1"),("xavier","init-xavier"),("normal","init-normal"),("zeros","init-zeros")):
                r=by_id[eid]; st=r["summary"]
                lines.append(f"| {init} / `{eid}` | {', '.join(f'{v:.5f}' for v in st['activation_std'])} | {st['step0_loss']:.6f} |")
            lines.append("Zeros khiến hidden activation và gradient lớp ẩn bằng 0; chỉ bias đầu ra học được tần suất lớp. "
                "Normal std=0,01 dễ làm tín hiệu nhỏ qua nhiều lớp. He có Var=2/fan_in để bù ReLU, Xavier dùng "
                "2/(fan_in+fan_out). Mạng chỉ hai lớp ẩn nên chưa đại diện cho hiện tượng suy giảm/bùng nổ ở mạng rất sâu.")
        lines.append(evidence(group))
    lines += ["", "## 4. Đánh giá cuối trên eval",
        f"Khóa cấu hình `{final['cfg']['exp_id']}`, seed 1 bằng val F1={final['summary']['val_macro_f1']:.6f}, "
        f"checkpoint epoch {final['summary']['best_epoch']}, trước khi gọi evaluate.py. Không chỉnh cấu hình sau eval.",
        "", "| Cấu hình | eval accuracy | eval macro-F1 |", "|---|---:|---:|",
        f"| Baseline `base-s1` | {be['accuracy']:.6f} | {be['macro_f1']:.6f} |",
        f"| Cuối `{final['cfg']['exp_id']}` | {ev['accuracy']:.6f} | {ev['macro_f1']:.6f} |",
        f"Cải thiện eval F1={ev['macro_f1']-be['macro_f1']:+.6f}. Chênh val–eval của mô hình cuối "
        f"={final['summary']['val_macro_f1']-ev['macro_f1']:+.6f}. Cố định seed 1 cho file nộp, không chọn seed theo eval. "
        "Seed 2/3 của cùng cấu hình (`final-s2`, `final-s3`) chỉ đo nhiễu; cấu hình đã khóa trước khi chạy eval.",
        f"Eval F1 qua seed 1/2/3: baseline={base_eval.mean():.6f} ± {base_eval.std(ddof=1):.6f}, "
        f"cuối={final_eval.mean():.6f} ± {final_eval.std(ddof=1):.6f}; "
        f"cải thiện ghép cặp={paired.mean():+.6f} ± {paired.std(ddof=1):.6f}; "
        f"2σ của chênh lệch={2*paired.std(ddof=1):.6f}. "
        + ("Cải thiện trung bình vượt ngưỡng nhiễu tham khảo này. " if paired.mean()>2*paired.std(ddof=1) else "Chưa vượt ngưỡng nhiễu tham khảo này. ")
        + "Ba seed là mẫu nhỏ; đây chưa phải kiểm định thống kê đầy đủ. Mỗi số eval lấy từ script chính thức và có cột eval tương ứng trong Excel.",
        "", "### Phân tích lỗi theo lớp", "", "| Lớp | support | precision | recall | F1 |", "|---|---:|---:|---:|---:|"]
    for c in ev["per_class"]:
        lines.append(f"| {c['cls']} | {c['support']} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1']:.6f} |")
    cm=np.array(ev["confusion_matrix"])
    worst=min(ev["per_class"],key=lambda c:c["f1"])
    row=cm[worst["cls"]].copy(); row[worst["cls"]]=0; confused=int(row.argmax())
    lines.append(f"Lớp khó nhất {worst['cls']} (F1={worst['f1']:.6f}, support={worst['support']}); "
        f"nhầm sang lớp {confused} nhiều nhất ({row[confused]} mẫu). Mất cân bằng và đặc trưng địa hình chồng lấn "
        "là các giả thuyết, chưa kiểm chứng quan hệ nhân quả. Có thể thử weighted CE chỉ bằng val trong nghiên cứu tiếp theo.")
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap="Blues")
    for i in range(7):
        for j in range(7): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=7,color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set(xlabel="Predicted label",ylabel="True label",xticks=range(7),yticks=range(7),title="Final evaluation confusion matrix")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/"figures/eval-confusion.png",dpi=140); plt.close(fig)
    lines += ["", "![Ma trận nhầm lẫn](figures/eval-confusion.png)", "", "## 5. Câu hỏi dẫn dắt và hạn chế",
        "Optimizer tốt nhất và tác dụng dropout/clipping/AMP được trả lời bằng số ở mục 3; không có một kỹ thuật "
        "luôn thắng cho mọi lr, mức quá khớp hay phần cứng. AdamW khác Adam ở weight decay tách riêng, nhưng chưa thử ở đây.",
        "**Loss không giảm sau 2.000 bước: ba kiểm tra đầu tiên.** (1) Kiểm tra shape, dtype, nhãn 0..6, "
        "chuẩn hóa train-only và loss trước cập nhật để phát hiện dữ liệu/logits sai. (2) Tắt regularization rồi "
        "học thuộc 20 mẫu; nếu thất bại, kiểm tra zero_grad, optimizer và backward trước khi tăng năng lực model. "
        "(3) In gradient từng lớp, grad_norm trước clip và đường train/val; gradient 0 gợi ý ReLU chết/đứt graph, "
        "gradient lớn/NaN gợi ý lr hoặc precision, train thấp nhưng val tăng gợi ý quá khớp.",
        "Baseline và cấu hình cuối có 3 seed; các phép so sánh kỹ thuật còn lại chủ yếu 1 seed, 3 lr mỗi optimizer chưa phải tìm kiếm đầy đủ. "
        "Thời gian mỗi epoch gồm cả đánh giá train/val, chịu ảnh hưởng nhiệt độ và tác vụ nền; chưa benchmark nhiều lần. "
        "Best epoch theo val loss có thể khác epoch F1 cao nhất. Phép thử lr×10 không đảm bảo tạo NaN; kết quả ổn định "
        "cũng phải ghi nhận. Không kết luận nguyên nhân từ một quan sát tương quan.",
        "", "## 6. Phụ lục",
        f"{len(results)} lần chạy so sánh, mỗi lần một JSON/ảnh/dòng Excel. Thời gian huấn luyện và đánh giá epoch "
        f"tổng cộng {sum(sum(r['history']['epoch_time_s']) for r in results)/60:.2f} phút (không tính setup/health/xuất file). "
        "Có REPORT.md, experiments.xlsx, predictions_eval.csv, eval_result.json, baseline_eval_result.json, "
        "run_manifest.json, figures/, results/, code/. Trọng số chỉ lưu ở .work ngoài bộ nộp. "
        "Notebook có thể dùng lại lịch sử đã đo; REUSE_RESULTS=False huấn luyện lại trong thư mục mới."]
    outliers=[r for r in results if r['history']['epoch_time_s'] and
              max(r['history']['epoch_time_s'])>10*np.median(r['history']['epoch_time_s'])]
    for r in outliers:
        lines.insert(-2,f"Ngoại lệ thời gian `{r['cfg']['exp_id']}`: epoch dài nhất "
            f"{max(r['history']['epoch_time_s']):.3f}s, trung vị {np.median(r['history']['epoch_time_s']):.3f}s. "
            "Chưa xác định nguyên nhân gián đoạn; vẫn giữ mean và lịch sử gốc trong bảng/JSON, không dùng run này để kết luận tốc độ optimizer.")
    blocks=[]; table=[]
    for line in lines:
        if line.startswith("|"):
            table.append(line)
        else:
            if table: blocks.append("\n".join(table)); table=[]
            if line: blocks.append(line)
    if table: blocks.append("\n".join(table))
    (out/"REPORT.md").write_text("\n\n".join(blocks)+"\n",encoding="utf-8")
