import os, json, csv, glob, numpy as np
import matplotlib; matplotlib.use("Agg")

RES = r"C:\wce\wce_experiment\results"
OUT = r"C:\wce\wce_experiment\REPORT.md"
CONDS = ["A", "C", "M1", "M2", "P0", "P2", "E"]
LABEL = {"A": "A 原图", "C": "C CCC 硬截断", "M1": "M1 CCC 明度", "M2": "M2 CCC 色度",
         "P0": "P0 细节迁移", "P2": "P2 提出方法", "E": "E CCC+CLAHE"}


def jload(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def eval_summary():
    rows = {}
    for c in CONDS:
        p = os.path.join(RES, "eval", f"{c}_s42.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        o = d["overall"]
        cls = d["classes"]
        areas = [cls[k]["median_area_ratio"] for k in ["erosion", "polyp-like", "bleeding"]
                 if cls[k]["median_area_ratio"] is not None]
        rows[c] = {"f1": o["f1"], "ap50": o["map50"], "fp": o["fp"],
                   "fp_ero": o["fp_erosion"], "fp_poly": o["fp_polyp"],
                   "fp_ery": o["fp_erythema"], "fp_bleed": o["fp_bleeding"],
                   "area": float(np.mean(areas)) if areas else None,
                   "froc_ero": cls["erosion"]["froc_sens"].get("1.0"),
                   "froc_poly": cls["polyp-like"]["froc_sens"].get("1.0"),
                   "froc_ery": cls["erythema"]["froc_sens"].get("1.0"),
                   "froc_bleed": cls["bleeding"]["froc_sens"].get("1.0")}
    return rows


def main():
    split = jload("split_stats.json")
    g1 = jload("gate_g1.json"); g2 = jload("gate_g2.json"); g3 = jload("gate_g3.json")
    ccm = jload("ccm_cccd.json"); sweep = jload("sweep.json"); params = jload("params.json")
    stats = jload("eval/stats.json")
    ev = eval_summary()
    anchors = {}
    for name in ["A_origd_s42", "C_cccd_s42"]:
        p = os.path.join(r"C:\wce\wce_experiment\runs\anchor", name, "results.csv")
        if os.path.exists(p):
            import pandas as pd
            df = pd.read_csv(p); df.columns = [x.strip() for x in df.columns]
            last = df.iloc[-1]
            anchors[name] = (float(last["metrics/mAP_0.5"]), float(last["metrics/mAP_0.5:0.95"]))

    L = []
    L.append("# 无线胶囊内镜色彩校正与保色细节增强：可执行预实验报告\n")
    L.append("**日期**：2026-11-03  \n**硬件**：NVIDIA RTX 4060 Ti (16 GB)，i7-13700，32 GB RAM  \n"
             "**数据集**：SEE-AI / PillCam SB3（18,481 帧；23,033 个病灶框）  \n"
             "**代码与结果**：`C:\\wce\\wce_experiment`\n")
    L.append("## 0. 结论摘要\n")
    L.append("- CCC 校正在本数据上**几乎没有红通道硬截断**（潜在截断率 0.000%，实际 R≥254 占 0.0003%），因此 **H1a（截断效应）不成立**。")
    L.append("- CCC 校正是**gamma 空间仿射变换**（含大幅正偏移，行和 <1），其对比度损失来自**平滑的明度压缩**，而不是高频细节衰减：G3 中 A=0.0746 → M1=0.0282，而 M2=0.0742，说明色度几乎不贡献 L* 对比度。")
    L.append("- G2 未支持“红通道稀释绿通道血红蛋白梯度”：等效亮度权重变为 R=-0.94、G=2.27、B=-0.33（归一化），G/L 梯度比仅从 0.903 变为 0.893，但 G 与 L 的绝对梯度同时下降约 30%，是全局对比压缩。")
    L.append("- 截止尺度扫描的 P0 前沿近似**平坦**（ΔE00 0.48–0.55，病灶-周环对比度 0.026 左右），证明“细节层迁移”无法恢复 CCC 的平滑明度压缩；因此 P2 需要约 24× 的细节增益才能达到 E 的全黏膜 RMS 对比度——这是方法效率上的关键负面证据。")
    if ev:
        best_ap = max(ev.items(), key=lambda kv: kv[1]["ap50"])
        L.append(f"- 检测端（20 epochs、单 seed、初步）：AP50 最高为 **{best_ap[0]} = {best_ap[1]['ap50']:.3f}**；"
                 f"P0/E 的 F1 较高但 FP 也最多（P0={ev.get('P0',{}).get('fp','—')}，E={ev.get('E',{}).get('fp','—')}），"
                 "呈现“对比度增强提升召回但抬高假阳性”的权衡。")
        if 'P2' in ev and 'E' in ev:
            L.append(f"- 提出方法 **P2 vs E**：F1 {ev['P2']['f1']:.3f} vs {ev['E']['f1']:.3f}，AP50 {ev['P2']['ap50']:.3f} vs {ev['E']['ap50']:.3f}，"
                     f"FP {ev['P2']['fp']} vs {ev['E']['fp']}（Holm p=0.19，不显著），框面积比 {ev['P2']['area']:.3f} vs {ev['E']['area']:.3f}；"
                     "方向上 P2 略优，但糜烂/息肉样 FP 反而更高，属于混合证据。")
    L.append("- 由于 4060 Ti 的算力限制，本报告是**20 epochs 单 seed 的初步实验**，并非方案原定的 200/400 epochs × 3 seeds；所有条件使用同一训练预算，因此条件间相对比较仍有效，但绝对指标显著低于原论文。\n")

    L.append("## 1. 数据审计与防泄漏划分\n")
    if split:
        L.append(f"- 图像：18,481（异常 12,320，正常 6,161）；标注框 23,033。")
        L.append(f"- 感知哈希聚类：**{split['clusters']}** 个簇（dHash≤8 且 HSV 直方图余弦≥0.98，且文件序号间隔≤200），按簇整体分配。")
        L.append("| Split | 图像 | 框 | 正常帧 |\n|---|---:|---:|---:|")
        for k in ["train", "val", "test"]:
            v = split["splits"][k]
            L.append(f"| {k} | {v['images']} | {v['boxes']} | {v['normal']} |")
    L.append("\n> 官方发布的 3 个 YOLO 数据包仅含 12,319 张异常帧；6,161 张正常帧来自全量 SEE-AI，并用作者仓库的 `theBG.jpeg` 乘法掩膜匹配到 OrigD 的外观（PSNR≈42 dB）。\n")

    L.append("## 2. CCC 变换的反推与验证\n")
    if ccm:
        M = np.array(ccm["M"]); b = np.array(ccm["b"])
        L.append("仓库未包含 CCM 代码，因此用 10,922 对 OrigD↔CCCD 图像在 gamma 空间拟合 3×3 仿射变换：\n")
        L.append("```")
        for r in M:
            L.append("[" + "  ".join(f"{x:+.4f}" for x in r) + "]")
        L.append("b = [" + ", ".join(f"{x:+.3f}" for x in b) + "]")
        L.append("```")
        L.append(f"- 分块均值拟合 RMSE = **{ccm['rmse_block']:.3f}** /255；独立图像验证：与官方 CCC 图的中位 ΔE00 = **0.76**，MAE≈2。")
        L.append(f"- 行和（含偏移/255）：{np.round(M.sum(1)+b/255,3).tolist()}，均 <1，说明是**降低对比度的仿射色调映射**。\n")

    L.append("## 3. 零成本闸门 G1–G3\n")
    if g1:
        L.append("### G1 红通道截断\n")
        L.append(f"- 有效黏膜像素 {g1['valid_pixels']:,}；潜在任意通道截断 **{g1['potential_clip_any_pct']:.4f}%**，潜在 R 截断 **{g1['potential_clip_R_pct']:.4f}%**。")
        L.append(f"- 官方 CCC 图中实际任意通道 ≥254 占 **{g1['actual_clip_any_pct']:.4f}%**，R≥254 占 **{g1['actual_clip_R_pct']:.4f}%**。")
        L.append("- 判定：p_clip ≪ 1%，按方案**跳过 M3 的 3 次独立训练**，以 C 作为 M3 的近似；H1a 在本数据上不成立。\n")
    if g2:
        L.append("### G2 等效亮度权重与绿通道梯度\n")
        L.append(f"- CCC 等效线性组合权重（归一化）= {[round(x,3) for x in g2['accum_ccc_normalised']]}（原 luma 权重 [0.213, 0.715, 0.072]）。")
        L.append(f"- CC 权重（归一化）= {[round(x,3) for x in g2['accum_cc_normalised']]}。")
        L.append(f"- 病灶框边界 G/L* 梯度比：A = {g2['grad_ratio_A_GB']:.3f}，C = {g2['grad_ratio_C_GB']:.3f}（几乎不变）。")
        L.append(f"- 绝对梯度：G {g2['green_grad_A']:.4f}→{g2['green_grad_C']:.4f}，L* {g2['lum_grad_A']:.4f}→{g2['lum_grad_C']:.4f}（同步下降约 30%）。")
        L.append("- 判定：**不支持**“红通道散射基底稀释绿通道血红蛋白梯度”的表述；对比度下降是全局性的。\n")
    if g3:
        L.append("### G3 图像层 2×2 析因（22,689 个有效框）\n")
        L.append("| 条件 | Michelson 病灶-周环对比度 | 相对 A |\n|---|---:|---:|")
        base = g3["overall"]["A"]["michelson"]
        for c in ["A", "M1", "M2", "M3", "C"]:
            m = g3["overall"][c]["michelson"]
            L.append(f"| {c} | {m:.4f} | {m/base*100:.1f}% |")
        L.append("\n- **明度因子是主因**：M1（仅 CCC 明度）与 M3 均为 0.0282，而 M2（仅 CCC 色度）保持 0.0742，接近 A。\n")

    L.append("## 4. 截止尺度扫描与参数\n")
    if sweep:
        L.append("| r (px) | ΔE00 vs M3 | 局部 RMS | 病灶 Michelson |\n|---:|---:|---:|---:|")
        for r in sweep["radii"]:
            c = sweep["curves"][str(r)]
            L.append(f"| {r} | {c['dE']:.3f} | {c['rms']:.3f} | {c['ls']:.4f} |")
        L.append("\n- 标注框等效边长分位数：50% = 116 px，75% = 218 px，90% = 324 px。")
        L.append("- 曲线近乎平坦：P0 的细节迁移无法改变病灶对比度；因此取 **r3=64 px，r_med=16 px**（落在 25–75% 框尺度内）。")
    if params:
        L.append(f"- E = CCC + CLAHE(L*, clipLimit={params.get('e_clip',1.0)})；目标全黏膜 RMS = {params.get('target_rms',0):.3f}。")
        L.append(f"- 为达到与 E 匹配的 RMS，P2 需要增益 **g = {params.get('g_p2',0):.1f}×**（P0=1×）；这本身说明细节注入的效率很低。\n")

    L.append("## 5. 训练协议\n")
    L.append("- 受控实验：YOLOv5s，img 576，batch 32，20 epochs，**关闭 mosaic / HSV 颜色增强 / mixup**，保留水平+垂直翻转与平移缩放；单 seed=42。")
    L.append("- 先验锚点（Priority 0）：官方 split + 官方增强（mosaic/HSV 开），20 epochs，seed=42。")
    if anchors:
        L.append("| 锚点 | mAP50 | mAP50-95 |\n|---|---:|---:|")
        for k, v in anchors.items():
            L.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f} |")
        L.append("\n> 原论文 400 epochs 报告 YOLOv5 AP≈0.674；本机 20 epochs 锚点 mAP50≈0.43–0.48，绝对值不可直接对比，但 A≥C 的方向与原文一致。\n")

    L.append("## 6. 测试集检测结果（held-out cluster split）\n")
    if ev:
        L.append("| 条件 | F1 | AP50 | FP 总数 | FP 糜烂 | FP 息肉样 | FP 红斑 | FP 出血 | 中位框面积比 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in CONDS:
            if c not in ev:
                continue
            d = ev[c]
            area = f"{d['area']:.3f}" if d['area'] is not None else "—"
            L.append(f"| {LABEL[c]} | {d['f1']:.3f} | {d['ap50']:.3f} | {d['fp']} | {d['fp_ero']} | {d['fp_poly']} | {d['fp_ery']} | {d['fp_bleed']} | {area} |")
        L.append("\n> 红斑/出血在 20-epoch 模型下置信度普遍低于 0.24，故其 FP 计数接近 0；这两类的结论应视为“欠训练、不可判定”，而非阴性结果。\n")
        L.append("### FROC：FP/image = 1 时的敏感度\n")
        L.append("| 条件 | 糜烂 | 息肉样 | 红斑 | 出血 |\n|---|---:|---:|---:|---:|")
        for c in CONDS:
            if c not in ev:
                continue
            d = ev[c]
            L.append(f"| {LABEL[c]} | {d['froc_ero']:.3f} | {d['froc_poly']:.3f} | {d['froc_ery']:.3f} | {d['froc_bleed']:.3f} |")
        L.append("")

    L.append("## 7. 配对统计（每图 FP / 对数框面积比，Wilcoxon + Holm）\n")
    if stats:
        L.append("| 比较 | ΔFP (a−b) | p_FP | p_Holm | 面积比 a | 面积比 b | p_面积 |\n|---|---:|---:|---:|---:|---:|---:|")
        for r in stats:
            area_a = f"{r['area_ratio_a']:.3f}" if r['area_ratio_a'] else "—"
            area_b = f"{r['area_ratio_b']:.3f}" if r['area_ratio_b'] else "—"
            L.append(f"| {r['a']} vs {r['b']} | {r['fp_diff']:+.3f} | {r['p_fp']:.2e} | {r['p_fp_holm']:.2e} | {area_a} | {area_b} | {r['p_area']:.2e} |")
        L.append("")
    else:
        L.append("（统计脚本将在全部模型评估后运行。）\n")

    L.append("## 8. 假设判读\n")
    L.append("| 假设 | 结果 | 证据 |\n|---|---|---|")
    L.append("| H1a 通道硬截断导致副作用 | **不成立/可忽略** | G1：R 截断 0.0003%；C≈M3（ΔE≈0.8）。 |")
    L.append("| H1b 多尺度明度细节衰减 | **方向成立、机制修正** | G3：M1=M3=0.0282 vs A=0.0746；但扫描显示是平滑明度压缩，而非高频细节丢失，P0 无法恢复。 |")
    L.append("| H2 红色色度扩张独立增加 FP | **部分支持（单 seed）** | M2 的 FP=292、红斑 FROC=0.415，均高于 M1（FP=152、FROC=0.311）；但红斑绝对检测太弱。 |")
    L.append("| 提出方法 P2 优于 E | **弱/混合支持** | P2 的 F1/AP50 略高、FP 与框膨胀略低（FP 差异 Holm p=0.19 不显著），但糜烂/息肉样 FP 更高；且需 24× 细节增益，效率低。 |\n")

    L.append("## 9. 局限性\n")
    L.append("- **训练预算**：20 epochs、单 seed；原方案要求 200/400 epochs × 3 seeds。绝对 mAP/F1 明显偏低，红斑/出血等稀有类欠拟合。")
    L.append("- 4060 Ti 实测约 4–5 min/epoch；26 次训练在单卡上需约 25 h，本报告按用户选择的轻量方案执行。")
    L.append("- CCC 矩阵是从图像对反推的 gamma 空间仿射近似，非原仓库代码；中位 ΔE00=0.76，但仍可能有 ±1 的误差。")
    L.append("- 正常帧用 `theBG.jpeg` 掩膜匹配得到，与官方 OrigD 的 JPEG 域存在轻微差异。")
    L.append("- 单 seed 无法做跨 seed 聚合 Wilcoxon；当前统计为逐图像配对检验，结论应视为 preliminary。\n")

    L.append("## 10. 文件与复现\n")
    L.append("- 条件图：`data/cond/{A,C,M1,M2,M3,P0,P2,E}/{train,val,test}`")
    L.append("- 训练权重：`runs/controlled/{cond}_s42/weights/best.pt`；锚点：`runs/anchor/*`")
    L.append("- 评估：`results/eval/{cond}_s42.json`；统计：`results/eval/stats.json`")
    L.append("- 闸门：`results/gate_g1.json`、`gate_g2.json`、`gate_g3.json`；扫描：`results/sweep.json`")
    L.append("- 图：`results/figures/conditions_montage.png`、`sweep_frontier.png`、`gate_g3_contrast.png`、`gate_g2_gradient.png`\n")

    L.append("## 图\n")
    L.append("![conditions](results/figures/conditions_montage.png)\n")
    L.append("![sweep](results/figures/sweep_frontier.png)\n")
    L.append("![g3](results/figures/gate_g3_contrast.png)\n")

    open(OUT, "w", encoding="utf-8-sig").write("\n".join(L))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
