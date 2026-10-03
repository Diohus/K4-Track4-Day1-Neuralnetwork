"""Hypotheses fixed before training; all comparisons use the same split."""
HYPOTHESES = {
    "baseline": "SGD momentum với lr phù hợp sẽ vượt đoán lớp đa số; khác seed có thể làm metric dao động.",
    "loss": "MSE trên logits và one-hot có thang đo khác CE; có thể hội tụ chậm hoặc khác về macro-F1. So bằng metric, không so loss tuyệt đối.",
    "optimizer": "Adam điều chỉnh bước theo từng tham số nên có thể hội tụ nhanh hơn SGD momentum, nhưng phải thử lr riêng cho mỗi optimizer.",
    "hparam": "M-wide tăng năng lực biểu diễn, có thể tăng macro-F1 nhưng tốn thời gian và bộ nhớ hơn; chưa chắc có lợi nếu chưa hội tụ.",
    "dropout": "Dropout 0.3 có thể giảm gap nếu baseline quá khớp; nếu train và val vẫn cùng giảm thì có thể làm chậm học.",
    "clipping": "Clip phải kích hoạt để có tác dụng. Ở lr cao, giới hạn chuẩn gradient có thể giảm dao động nhưng không bảo đảm cứu được huấn luyện.",
    "amp": "FP16 có thể giảm bộ nhớ, nhưng mạng nhỏ và GPU này có thể không nhanh hơn; chất lượng kỳ vọng gần FP32.",
    "init": "Zeros giữ đối xứng và ReLU tại 0 chặn gradient lớp ẩn. Xavier/normal thay đổi phương sai kích hoạt và tốc độ học so với He.",
    "final": "Cấu hình đã chọn bằng validation có thể dao động giữa seed; lặp seed 2 và 3 để đo nhiễu, không chọn lại theo eval.",
}

def search_configs(base):
    configs = []
    for opt, rates in (("sgd_momentum", (0.01, 0.05, 0.1)), ("adam", (0.0003, 0.001, 0.003))):
        for lr in rates:
            configs.append({**base, "exp_id": f"search-{opt}-lr{lr:g}", "group": "optimizer",
                "description": f"Validation LR search: {opt}, lr={lr:g}", "optimizer": opt, "lr": lr,
                "hypothesis": HYPOTHESES["optimizer"],
                "notes": "20 epochs; optimizer and its learning rate calibrated together; fixed seed 1."})
    return configs

def topic_configs(base, clip_norm, cuda):
    variants = [
        ("loss-mse", "loss", {"loss": "mse"}),
        ("wide", "hparam", {"hidden": (512,256)}),
        ("drop-0.3", "dropout", {"dropout": 0.3}),
        ("clip-normal", "clipping", {"clip_norm": clip_norm}),
        ("highlr-no-clip", "clipping", {"lr": base["lr"]*10}),
        ("highlr-clip", "clipping", {"lr": base["lr"]*10,"clip_norm": clip_norm}),
        ("init-zeros", "init", {"init": "zeros"}),
        ("init-normal", "init", {"init": "normal"}),
        ("init-xavier", "init", {"init": "xavier"}),
    ]
    if cuda: variants.append(("amp-fp16", "amp", {"precision": "fp16"}))
    return [{**base, **changes,"exp_id": exp_id,"group": group,
             "description": f"Controlled {group} comparison", "hypothesis": HYPOTHESES[group],
             "notes": "High-lr clipping pair changes lr relative to baseline; clip is the only difference within the pair."
                      if exp_id.startswith("highlr") else "One factor changed relative to baseline; fixed seed 1."}
            for exp_id,group,changes in variants]
