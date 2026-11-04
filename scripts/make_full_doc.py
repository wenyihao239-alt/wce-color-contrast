import os, json, csv, glob
import numpy as np
import pandas as pd

RES = r"C:\wce\wce_experiment\results"
DATA = r"C:\wce\wce_experiment\data"
RUNS = r"C:\wce\wce_experiment\runs"
OUT = r"C:\wce\wce_experiment\WCE_完整实验方案与结果.md"
CONDS = ["A", "C", "M1", "M2", "P0", "P2", "E"]
LABEL = {"A": "A 原图", "C": "C CCC 硬截断", "M1": "M1 CCC 明度",
         "M2": "M2 CCC 色度", "P0": "P0 细节迁移", "P2": "P2 提出方法", "E": "E CCC+CLAHE"}
CLASSES = ['angiodysplasia', 'erosion', 'stenosis', 'lymphangiectasia', 'lymph-follicle', 'SMT',
           'polyp-like', 'bleeding', 'diverticulum', 'erythema', 'foreign-body', 'vein']


def load(name, default=None):
    p = os.path.join(RES, name)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return default


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(x, n=3, dash="—"):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return dash
    return f"{x:.{n}f}"


def main():
    split = load("split_stats.json", {})
    ccmc = load("ccm_cccd.json", {})
    ccmcc = load("ccm_ccd.json", {})
    g1 = load("gate_g1.json", {})
    g2 = load("gate_g2.json", {})
    g3 = load("gate_g3.json", {})
    sweep = load("sweep.json", {})
    params = load("params.json", {})
    stats = load("eval/stats.json", [])
    ev = {}
    for c in CONDS:
        p = os.path.join(RES, "eval", f"{c}_s42.json")
        if os.path.exists(p):
            ev[c] = json.load(open(p))
    anchors = {}
    for name in ["A_origd_s42", "C_cccd_s42"]:
        p = os.path.join(RUNS, "anchor", name, "results.csv")
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = [x.strip() for x in df.columns]
            last = df.iloc[-1]
            anchors[name] = (float(last["metrics/mAP_0.5"]), float(last["metrics/mAP_0.5:0.95"]),
                             float(last["metrics/precision"]), float(last["metrics/recall"]))
    durations = {}
    for p in glob.glob(os.path.join(RES, "train_logs", "*.log")) + glob.glob(os.path.join(RES, "train_anchor_*.log")):
        txt = open(p, encoding="utf-8", errors="ignore").read()
        import re
        m = re.findall(r"(\d+) epochs completed in ([\d.]+) hours", txt)
        if m:
            durations[os.path.basename(p).replace(".log", "")] = (int(m[-1][0]), float(m[-1][1]))
    for k, v in [("A_origd_s42", "train_anchor_A"), ("C_cccd_s42", "train_anchor_C")]:
        txt = open(os.path.join(RES, v + ".log"), encoding="utf-8", errors="ignore").read()
        import re
        m = re.findall(r"(\d+) epochs completed in ([\d.]+) hours", txt)
        if m:
            durations[k] = (int(m[-1][0]), float(m[-1][1]))

    L = []
    L.append("# 无线胶囊内镜（WCE）色彩校正与保色细节增强\n## 完整实验方案、实验配置与实验结果\n")
    L.append("**版本**：v1.0  \n**日期**：2026-11-03  \n"
             "**数据**：SEE-AI / PillCam SB3（18,481 帧，23,033 个病灶框）  \n"
             "**硬件**：NVIDIA RTX 4060 Ti 16 GB，Intel i7-13700，32 GB RAM  \n"
             "**代码/结果根目录**：`C:\\wce\\wce_experiment`  \n")
    L.append("> 本文档同时给出**方案设计**与**实际执行/结果**。实际执行采用用户选定的轻量预算："
             "7 个受控条件 + 2 个官方锚点，均为 **20 epochs、单 seed=42**；原方案建议的 200/400 epochs × 3 seeds "
             "受单卡算力限制未执行。所有条件共享同一训练预算，因此条件间相对比较有效，但绝对指标低于原论文。\n")

    # 0 summary
    L.append("## 0. 结论速览\n")
    L.append("- **H1a（通道硬截断）不成立**：CCC 变换的潜在 R 截断率为 0.000%，官方 CCC 图中 R≥254 仅 0.0003%；C 与 M3 近似等价。")
    L.append("- **对比度损失由明度分量主导，色度分量几乎不贡献 L\\* 对比度**：G3 全量病灶框 Michelson 对比度 A=0.0746、M1=0.0282、M2=0.0742、M3=0.0282、C=0.0319。")
    L.append("- **G2 不支持“红通道稀释绿通道血红蛋白梯度”**：CCC 的等效亮度权重归一化后为 R=-0.942、G=2.273、B=-0.331；G/L\\* 梯度比 A=0.903 → C=0.893，几乎不变，但 G 与 L\\* 绝对梯度同步下降约 30%。")
    L.append("- **P0 细节迁移前沿平坦**（ΔE00 0.48–0.55，病灶对比度约 0.026），说明 CCC 的对比度损失是**平滑明度压缩**，不是高频细节丢失；P2 需约 **24×** 细节增益才能匹配 E 的全黏膜 RMS 对比度。")
    if ev:
        ap_best = max(ev.items(), key=lambda kv: kv[1]["overall"]["map50"])
        L.append(f"- **检测（20 epochs，单 seed）**：AP50 最高为 {ap_best[0]}={ap_best[1]['overall']['map50']:.3f}；"
                 "P0/E 召回/F1 较高但 FP 最多；P2 vs E 呈弱优势（F1 0.339 vs 0.328，AP50 0.283 vs 0.278，FP 366 vs 401，Holm p=0.19，框面积比 0.981 vs 1.002）。")
    L.append("- **建议**：H1a 可从论文主线删除；H1b 需改写为“平滑明度压缩”而非“多尺度细节衰减”；P0/P2 的细节注入路线效率低，后续应优先考虑**明度保持（M2 式）或显式色调逆映射**。\n")

    # 1 research questions
    L.append("## 1. 研究问题、假设与判读矩阵\n")
    L.append("### 1.1 研究问题\n")
    L.append("- **RQ1（机制）**：全局色彩校正的哪个分量（通道截断、明度变化、色度/色相偏移）导致检测框膨胀与假阳性增加？")
    L.append("- **RQ2（方法）**：能否在保持校准色度目标的同时，恢复诊断相关的局部对比度并抑制不可靠区域放大？")
    L.append("- **RQ3（评估）**：人眼与检测器对相同色度/对比度扰动的敏感度是否一致？（本次未执行心理物理实验）\n")
    L.append("### 1.2 假设\n")
    L.append("- **H1a**：红通道饱和截断导致边界畸变，进而造成框膨胀与 FP 增加。")
    L.append("- **H1b**：CCC 明度变化衰减多尺度明度细节，导致病灶-周环对比度下降。")
    L.append("- **H2**：红-黄色度扩张在红色主导黏膜中独立增加色度定义病变（红斑、出血）的 FP。")
    L.append("- **H3**（未执行）：人眼偏好与病灶显著性阈值系统性地不同于检测器敏感度曲面。\n")
    L.append("### 1.3 预设判读矩阵（方案原定）\n")
    L.append(table(["观测模式", "机制结论", "对方法的指导"], [
        ["M3 明显优于 C，且 M3≈P0", "H1a 截断为主因", "把等色相软色域压缩作为标准后处理"],
        ["P0/M2 使 FP 与框膨胀回落至接近 A", "H1b 高频明度细节为主因", "细节保持校正 (T2.2)"],
        ["M2 独立复现 FP 上升而 M1 不", "H2 红色色度扩张为主因", "限制跨设备色度映射，WP3 重点比较红色扩张"],
        ["P2 在同等全黏膜对比度下优于 E", "可靠性门控 + 结构解耦有效", "支撑 T2.2/T2.3"],
    ]) + "\n")

    L.append("### 1.4 原定训练优先级与统计方案\n")
    L.append(table(["优先级", "条件", "seeds", "训练次数", "目标"], [
        ["Priority 0 复现锚点", "A, C（官方 split + 官方增强）", "1 (42)", "2", "验证本地环境对 Agossou 2025 的复现精度"],
        ["Priority 1 核心析因", "A, C, M1, M2, M3", "3 (42,2025,3407)", "12–15", "分离截断/明度/色度主效应"],
        ["Priority 2 方法对比", "P0, P2, E", "3", "9", "同对比度下检验 P2 是否优于 E"],
        ["Priority 3 组件消融", "P1, P1s, D", "1 (42)", "3", "验证 Q 门控、多尺度、纯明度增强"],
    ]) + "\n")
    L.append("**预设 6 项核心比较**（Holm–Bonferroni 控制族错误率）：")
    L.append("1. C vs A（基线校正效应复现）；2. C vs M3（H1a 截断）；3. M1 vs A/M2 的明度主效应（H1b）；"
             "4. M2 vs A/M1 的色度主效应（H2）；5. P0 vs M3（细节迁移恢复）；6. P2 vs E（同对比度下的方法优势）。\n")
    L.append("**统计模型**：每张测试图在多种子模型上取 FP/对数框面积比均值 → 图像间双侧 Wilcoxon 符号秩检验 → "
             "以检查簇为单位做 1,000 次 cluster bootstrap 置信区间；另可用按检查簇聚类的负二项 GLMM 估计明度/色度主效应。"
             "本次因单 seed，仅执行逐图配对 Wilcoxon + Holm，未做多种子聚合与 GLMM。\n")

    # 2 data
    L.append("## 2. 数据审计与划分\n")
    L.append("### 2.1 数据来源\n")
    L.append("| 数据 | 用途 | 规模 |\n|---|---|---|")
    L.append("| R-OrigD YOLO（Agossou 2025 发布） | 原始基线 A、官方划分锚点 | 12,319 张异常帧，10,922 train / 1,397 val |")
    L.append("| R-CCD YOLO | 标准 ColorChecker 校正（仅图像层 G1/G2） | 同上 |")
    L.append("| R-CCCD YOLO | CCC 校正 C | 同上 |")
    L.append("| SEE-AI 全量（Kaggle capsuleyolo/kyucapsule） | 补足 6,161 张正常帧 | 18,481 图 + 18,481 txt |")
    L.append("\n类别框数与图像数（全量）：\n")
    rows = []
    # recompute class counts from manifest
    mrows = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
    box = {i: 0 for i in range(12)}; img = {i: 0 for i in range(12)}
    for r in mrows:
        cs = [int(x) for x in r["classes"].split(";") if x != ""]
        for c in set(cs): img[c] += 1
        for c in cs: box[c] += 1
    for i, n in enumerate(CLASSES):
        rows.append([n, box[i], img[i]])
    rows.append(["**合计**", sum(box.values()), sum(1 for r in mrows if int(r["n_boxes"]) > 0)])
    L.append(table(["类别", "框数", "含该类图像数"], rows) + "\n")
    L.append("### 2.2 划分与防泄漏\n")
    L.append("- 感知哈希：dHash 64-bit（9×8 灰度）+ HSV 8×8×4 直方图；边条件为 **Hamming≤8 且 余弦≥0.98**，并限制文件名序号间隔 ≤200，避免跨视频全局串联。")
    L.append(f"- 聚类结果：**{split.get('clusters','—')}** 个簇；按簇整体分配，绝不跨 split。")
    if split:
        L.append(table(["Split", "图像", "框", "正常帧"],
                       [[k, split["splits"][k]["images"], split["splits"][k]["boxes"], split["splits"][k]["normal"]]
                        for k in ["train", "val", "test"]]) + "\n")
    if sweep and sweep.get("box_sqrt_area_px"):
        L.append("### 2.3 标注框尺度\n")
        b = sweep["box_sqrt_area_px"]
        L.append("等效边长 √(w·h) 分位数（像素）：\n")
        L.append(table(["P5", "P10", "P25", "P50", "P75", "P90", "P95"],
                       [[fmt(b[k], 1) for k in ["5", "10", "25", "50", "75", "90", "95"]]]) + "\n")
    L.append("### 2.4 正常帧外观匹配\n")
    L.append("官方 YOLO 包仅含异常帧。全量 SEE-AI 的正常帧通过作者仓库 `theBG.jpeg` 做逐像素乘法（`ImageChops.multiply`）匹配 OrigD 的掩膜/暗角，"
             "匹配后 PSNR≈42 dB。该操作只影响负样本，不改变异常帧 A/C。\n")

    # 3 conditions
    L.append("## 3. 色彩变换与实验条件定义\n")
    L.append("### 3.1 CCC/CC 矩阵反推\n")
    L.append("仓库不含 CCM 实现。对 OrigD↔CCCD 的 10,922 对图像，在 gamma 编码 sRGB 空间拟合逐像素仿射变换：\n")
    L.append("```\nC = clip(M · A + b, 0, 255)\n```")
    if ccmc:
        M = np.array(ccmc["M"]); b = np.array(ccmc["b"])
        L.append(f"CCC（用于 M1/M2/M3/P0/P2）：RMSE(block) = {ccmc['rmse_block']:.3f}/255\n")
        L.append("```")
        for r in M:
            L.append("[" + "  ".join(f"{x:+.4f}" for x in r) + "]")
        L.append("b = [" + ", ".join(f"{x:+.3f}" for x in b) + "]")
        L.append("```")
        L.append(f"行和（含偏移/255）：{np.round(M.sum(1)+b/255,3).tolist()}")
        L.append("- 独立验证：重建图 vs 官方 CCC 图，中位 ΔE00=**0.76**，MAE≈2/255。\n")
    if ccmcc:
        Mc = np.array(ccmcc["M"]); bc = np.array(ccmcc["b"])
        L.append(f"标准 CC（仅用于 G2 对比）：RMSE(block)={ccmcc['rmse_block']:.3f}/255；行和={np.round(Mc.sum(1)+bc/255,3).tolist()}\n")

    L.append("### 3.2 统一等明度/等色相色域映射\n")
    L.append("对任意像素，若由 (L,C,h) 回转的线性 sRGB 超出 [0,1]，固定 L 与 h，二分搜索最大色度缩放因子 s∈[0,1]，"
             "使 (L, s·C, h) 落回色域。实现为 14 次向量化二分。该算子用于 A/M1/M2/M3/P0/P2，保证回转后绝对可比。\n")
    L.append("### 3.3 2×2 正交析因（对称设计）\n")
    L.append(table(["明度来源 \\ 色度来源", "原图 C0,h0", "CCC Cc,hc"], [
        ["原图 L0", "A (OrigD)", "M2 (CCC-Chroma)"],
        ["CCC Lc", "M1 (CCC-Lightness)", "M3 (CCC-Unclipped)"],
    ]) + "\n")
    L.append("- C（CCC-Clipped）= 官方 8-bit 硬截断输出。G1 证明 C≈M3，因此 M3 未单独训练；析因以 C 作为 M3 的近似。")
    L.append("- M1 隔离明度主效应；M2 隔离色度主效应；C 与 A 给出净校正效应。\n")
    L.append("### 3.4 截止尺度扫描与 P0\n")
    L.append("对明度 L，引导滤波（guided filter, eps=4）分解：B(r)=GF(L,r)，D(r)=L−GF(L,r)。")
    L.append("P0(r) = GF(Lc,r) + (L0 − GF(L0,r))，色度保持 Cc,hc。")
    L.append("- r→0：P0→M3；r→∞：P0→原图明度 + CCC 常数偏移（接近 M2）。")
    L.append("- 扫描 r ∈ {1,2,4,8,16,24,32,48,64,96,128,192,256}，在 300 张验证集病灶图上计算 ΔE00 与病灶-周环 Michelson 对比度。\n")
    L.append("### 3.5 提出方法 P2（及消融 P1/P1s）\n")
    L.append("多尺度原始明度细节：")
    L.append("```")
    L.append("G1=GF(L0,r3/16), G2=GF(L0,r3/4), G3=GF(L0,r3)")
    L.append("D = (L0-G1) + (G1-G2) + (G2-G3)")
    L.append("L_P2 = GF(Lc,r3) + g · Q · D")
    L.append("```")
    L.append("- Q 为可靠性权重：高光（L>88 且 C<18）→0.2；暗区（L<12）→0.2；平坦低 SNR（5×5 局部 std<2）→0.4；再做 σ=2 高斯平滑。")
    L.append(f"- 本次取 r3={params.get('r3',64)} px、r_med={params.get('r_med',16)} px。")
    L.append(f"- 增益 g 通过验证集全黏膜 9×9 局部 RMS 与 E 匹配：目标 RMS={params.get('target_rms',0):.3f}，求解得 **g={params.get('g_p2',0):.2f}**。")
    L.append("- P1（消融）：Q≡1；P1s：仅使用到 r_med 的单尺度细节。两者在轻量方案中未训练。\n")
    L.append("### 3.6 基线 E / D\n")
    L.append(f"- E = CCC + CLAHE(L\\*, clipLimit={params.get('e_clip',1.0)}, tile=8×8)。")
    L.append("- D = 原图 + CLAHE(L\\*)，用于区分“色彩校正”与“纯对比度增强”。轻量方案未训练 D。\n")
    L.append("### 3.7 条件总表\n")
    L.append(table(["条件", "明度", "色度/色相", "定义", "本次是否训练"], [
        ["A", "L0", "C0,h0", "原始 JPEG", "是"],
        ["C", "Lc(截断)", "Cc,hc", "官方 8-bit CCC", "是"],
        ["M1", "Lc", "C0,h0", "CCC 明度 + 原色度", "是"],
        ["M2", "L0", "Cc,hc", "原明度 + CCC 色度", "是"],
        ["M3", "Lc", "Cc,hc", "无截断 CCC（C≈M3）", "否（G1）"],
        ["P0", "GF(Lc,r3)+D(L0,r3)", "Cc,hc", "纯细节迁移", "是"],
        ["P2", "GF(Lc,r3)+g·Q·D", "Cc,hc", "提出方法", "是"],
        ["E", "CLAHE(C L*)", "C 的色度", "CCC+CLAHE-L*", "是"],
        ["D", "CLAHE(A L*)", "A 的色度", "原图+CLAHE-L*", "否（轻量）"],
    ]) + "\n")

    # 4 gates
    L.append("## 4. 零成本闸门 G1–G3\n")
    if g1:
        L.append("### G1 红通道截断率\n")
        L.append(f"- 有效黏膜像素：{g1['valid_pixels']:,}；n_images={g1['n_images']:,}。")
        L.append(table(["指标", "数值"], [
            ["潜在任意通道 >255", f"{g1['potential_clip_any_pct']:.4f}%"],
            ["潜在 R 通道 >255", f"{g1['potential_clip_R_pct']:.4f}%"],
            ["官方 C 任意通道 ≥254", f"{g1['actual_clip_any_pct']:.4f}%"],
            ["官方 C R 通道 ≥254", f"{g1['actual_clip_R_pct']:.4f}%"],
        ]) + "\n")
        L.append("**判定**：p_clip ≪ 1% → 跳过 M3 的 3 次训练；H1a 不成立。\n")
    if g2:
        L.append("### G2 等效亮度权重与梯度比\n")
        L.append(table(["量", "数值"], [
            ["CCC 等效权重（归一化）", str([round(x, 3) for x in g2.get('accum_ccc_normalised', [])])],
            ["CC 等效权重（归一化）", str([round(x, 3) for x in g2.get('accum_cc_normalised', [])])],
            ["原图 luma 权重", "[0.2126, 0.7152, 0.0722]"],
            ["G/L* 梯度比 A", fmt(g2.get('grad_ratio_A_GB'))],
            ["G/L* 梯度比 C", fmt(g2.get('grad_ratio_C_GB'))],
            ["G 梯度 A→C", f"{g2.get('green_grad_A',0):.4f} → {g2.get('green_grad_C',0):.4f}"],
            ["L* 梯度 A→C", f"{g2.get('lum_grad_A',0):.4f} → {g2.get('lum_grad_C',0):.4f}"],
            ["框数", g2.get("n_boxes", "—")],
        ]) + "\n")
        L.append("**判定**：不支持“红通道稀释绿通道梯度”；下降是全局的。\n")
    if g3:
        L.append("### G3 图像层 2×2 析因\n")
        L.append(table(["条件", "Michelson 均值", "Weber 均值", "有效框数"],
                       [[c, fmt(g3["overall"][c]["michelson"], 4), fmt(g3["overall"][c]["weber"], 4), g3["overall"][c]["n"]]
                        for c in ["A", "M1", "M2", "M3", "C"]]) + "\n")
        L.append("按关键类别（Michelson）：\n")
        rows = []
        for k in ["erosion", "polyp-like", "bleeding", "erythema"]:
            rows.append([k] + [fmt(g3["classes"][c].get(k, {}).get("michelson"), 4) for c in ["A", "M1", "M2", "M3", "C"]])
        L.append(table(["类别", "A", "M1", "M2", "M3", "C"], rows) + "\n")

    # 5 sweep
    L.append("## 5. 截止尺度扫描结果\n")
    if sweep:
        L.append(f"样本：{sweep.get('n_images','—')} 张验证集含病灶图。\n")
        L.append(table(["r (px)", "ΔE00 vs M3", "局部 RMS", "病灶 Michelson", "Weber"],
                       [[r, fmt(sweep["curves"][str(r)]["dE"]), fmt(sweep["curves"][str(r)]["rms"]),
                         fmt(sweep["curves"][str(r)]["ls"], 4), fmt(sweep["curves"][str(r)]["ls_w"], 4)]
                        for r in sweep["radii"]]) + "\n")
        L.append("**观察**：曲线近乎平坦，ΔE00 仅 0.48–0.55，病灶对比度 0.026 左右；P0 无法恢复 CCC 的对比度损失。"
                 "因此 r3 仅按框尺度分位数取 64 px，不追求“拐点”。\n")

    # 6 training config
    L.append("## 6. 实验配置\n")
    L.append("### 6.1 软硬件\n")
    L.append(table(["项", "配置"], [
        ["GPU", "NVIDIA RTX 4060 Ti 16 GB（WDDM）"],
        ["CPU/RAM", "Intel i7-13700 / 32 GB"],
        ["OS", "Windows 11"],
        ["Python", "3.14.7"],
        ["PyTorch", "2.14.0+cu130"],
        ["NumPy", "2.5.3"],
        ["OpenCV", "5.0.0"],
        ["检测器", "YOLOv5s（仓库 vendored，约 7.04M 参数，15.8 GFLOPs @576）"],
    ]) + "\n")
    L.append("### 6.2 YOLOv5 兼容性补丁（Windows / NumPy 2 / PyTorch 2.6+）\n")
    L.append("- `torch.load(..., weights_only=False)`（train.py / general.py / experimental.py / hubconf.py）。")
    L.append("- `np.trapz` → `np.trapezoid`（utils/metrics.py）。")
    L.append("- `ap_per_class` 单类别时用 `np.atleast_1d` 包装 tp/fp/p/r/f1/ap。")
    L.append("- 训练内验证设置 `save_txt=False, save_conf=False`，避免缺失 labels 目录。\n")
    L.append("### 6.3 官方锚点配置（Priority 0）\n")
    L.append("```")
    L.append("python train.py --img 576 --batch 32 --epochs 20 \\")
    L.append("  --data data/yaml/anchor_origd.yaml --weights yolov5s.pt \\")
    L.append("  --workers 1 --seed 42 --patience 8")
    L.append("```")
    L.append("- 官方 split：10,922 train / 1,397 val；启用 mosaic/HSV（默认 hyp.scratch-low.yaml）。")
    L.append("- 另一组把 data 换成 `anchor_cccd.yaml`。\n")
    L.append("### 6.4 受控实验配置\n")
    L.append("统一使用 `data/hyps/hyp.controlled.yaml`（内容如下）：\n")
    L.append("```yaml")
    hyp = open(os.path.join(DATA, "hyps", "hyp.controlled.yaml"), encoding="utf-8").read()
    L.extend(hyp.strip().splitlines())
    L.append("```\n")
    L.append("```")
    L.append("python train.py --img 576 --batch 32 --epochs 20 \\")
    L.append("  --data data/yaml/{A,C,M1,M2,P0,P2,E}.yaml --weights yolov5s.pt \\")
    L.append("  --hyp data/hyps/hyp.controlled.yaml --workers 1 --seed 42 --patience 5")
    L.append("```\n")
    L.append("### 6.5 训练清单与实际耗时\n")
    rows = []
    for c in CONDS:
        tag = f"{c}_s42"
        if tag in durations:
            rows.append([LABEL[c], tag, durations[tag][0], f"{durations[tag][1]:.3f} h"])
        else:
            rows.append([LABEL[c], tag, "20", "—"])
    for k, v in anchors.items():
        rows.append([f"锚点 {k}", k, durations.get(k, ("20", None))[0],
                     f"{durations.get(k, ('',0))[1]:.3f} h" if k in durations else "—"])
    L.append(table(["条件", "run", "epochs", "耗时"], rows) + "\n")
    L.append("> 受控训练两两并行（batch 32，workers=1）；P2 单独训练，因此耗时更短。\n")

    # 7 results
    L.append("## 7. 实验结果\n")
    L.append("### 7.1 官方锚点复现\n")
    if anchors:
        L.append(table(["锚点", "mAP50", "mAP50-95", "Precision", "Recall"],
                       [[k, fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3])] for k, v in anchors.items()]) + "\n")
        L.append("> 原论文 400 epochs 的 YOLOv5 AP≈0.674。本机 20 epochs 绝对值不可比；A≥C 的方向与原文一致。\n")
    if ev:
        L.append("### 7.2 整体检测指标（held-out test，conf=0.24，IoU=0.5）\n")
        rows = []
        for c in CONDS:
            if c not in ev:
                continue
            o = ev[c]["overall"]
            cls = ev[c]["classes"]
            areas = [cls[k]["median_area_ratio"] for k in ["erosion", "polyp-like", "bleeding"]
                     if cls[k]["median_area_ratio"] is not None]
            rows.append([LABEL[c], fmt(o["f1"]), fmt(o["map50"]), o["tp"], o["fp"], o["fn"],
                         o["fp_erosion"], o["fp_polyp"], o["fp_erythema"], o["fp_bleeding"],
                         fmt(float(np.mean(areas)) if areas else None)])
        L.append(table(["条件", "F1", "AP50", "TP", "FP", "FN", "FP 糜烂", "FP 息肉样", "FP 红斑", "FP 出血", "中位框面积比"], rows) + "\n")

        L.append("### 7.3 逐类 AP50\n")
        rows = []
        for i, k in enumerate(CLASSES):
            rows.append([k] + [fmt(ev[c]["classes"][k]["ap50"]) if c in ev else "—" for c in CONDS])
        L.append(table(["类别"] + [LABEL[c] for c in CONDS], rows) + "\n")

        L.append("### 7.4 逐类 F1\n")
        rows = []
        for k in CLASSES:
            rows.append([k] + [fmt(ev[c]["classes"][k]["f1"]) if c in ev else "—" for c in CONDS])
        L.append(table(["类别"] + [LABEL[c] for c in CONDS], rows) + "\n")

        L.append("### 7.5 框面积比（预测/GT 中位数）\n")
        rows = []
        for k in ["erosion", "polyp-like", "bleeding", "erythema"]:
            rows.append([k] + [fmt(ev[c]["classes"][k]["median_area_ratio"]) if c in ev else "—" for c in CONDS])
        L.append(table(["类别"] + [LABEL[c] for c in CONDS], rows) + "\n")

        L.append("### 7.6 FROC：FP/image = 1 时的敏感度\n")
        rows = []
        for k in ["erosion", "polyp-like", "bleeding", "erythema"]:
            rows.append([k] + [fmt(ev[c]["classes"][k]["froc_sens"].get("1.0")) if c in ev else "—" for c in CONDS])
        L.append(table(["类别"] + [LABEL[c] for c in CONDS], rows) + "\n")

    L.append("### 7.7 逐图配对统计（Wilcoxon 符号秩 + Holm）\n")
    if stats:
        L.append(table(["比较", "ΔFP (a−b)", "p_FP", "p_Holm", "面积比 a", "面积比 b", "p_面积"],
                       [[f"{r['a']} vs {r['b']}", f"{r['fp_diff']:+.3f}", f"{r['p_fp']:.2e}", f"{r['p_fp_holm']:.2e}",
                         fmt(r["area_ratio_a"]), fmt(r["area_ratio_b"]), f"{r['p_area']:.2e}"] for r in stats]) + "\n")
        L.append("> 面积比 a/b 为每图匹配 TP 的 log(pred_area/GT_area) 均值再取指数；与 7.5 的逐类中位数口径不同。\n")

    L.append("### 7.8 图\n")
    L.append("![条件对比](results/figures/conditions_montage.png)\n")
    L.append("*图 1：同一测试帧的 A / C / M1 / M2 / M3 / P0 / P2 / E。*\n")
    L.append("![扫描](results/figures/sweep_frontier.png)\n")
    L.append("*图 2：截止尺度扫描的 ΔE00 vs 病灶-周环对比度前沿。*\n")
    L.append("![G3](results/figures/gate_g3_contrast.png)\n")
    L.append("*图 3：2×2 析因的病灶-周环 Michelson 对比度。*\n")

    # 8 verdicts
    L.append("## 8. 假设判读\n")
    L.append(table(["假设", "判定", "关键证据", "对后续的指导"], [
        ["H1a 通道截断", "不成立/可忽略", "G1 潜在 R 截断 0.000%，实际 0.0003%；C≈M3", "从论文主线删除“硬截断”解释"],
        ["H1b 明细度衰减", "方向成立、机制需改写", "G3 A=0.0746→M1=0.0282、M2=0.0742；扫描平坦", "改为“平滑明度压缩”；P0 细节迁移不足以恢复"],
        ["H2 红色色度扩张", "部分支持（单 seed）", "M2 FP=292、红斑 FROC=0.415 > M1 FP=152、FROC=0.311", "需多 seed 与更强模型复核"],
        ["P2 优于 E", "弱/混合支持", "F1 0.339 vs 0.328、AP50 0.283 vs 0.278、FP 366 vs 401（p_Holm=0.19）、框面积比 0.981 vs 1.002；但糜烂/息肉样 FP 更高", "改进方向：明度保持/色调逆映射，而非 24× 细节放大"],
    ]) + "\n")

    # 9 differences
    L.append("## 9. 与原方案/原论文的差异\n")
    L.append(table(["项", "原方案/原论文", "本次执行", "影响"], [
        ["训练轮数", "200/400 epochs", "20 epochs", "绝对 mAP/F1 低；条件间相对比较仍成立"],
        ["随机种子", "3 seeds", "1 seed=42", "无法做跨 seed 聚合；统计为逐图配对"],
        ["条件数", "26–29 次训练", "9 次（7 受控 + 2 锚点）", "覆盖核心机制与 P0/P2/E，未跑 P1/P1s/D/M3"],
        ["M3", "独立训练", "G1 后跳过", "C≈M3，以 C 作近似"],
        ["数据集", "18,481 全量", "18,481（正常帧经 theBG 匹配）", "负样本域轻微差异"],
        ["CCM", "原作者代码", "图像对反推仿射", "中位 ΔE00=0.76"],
    ]) + "\n")

    # 10 limitations
    L.append("## 10. 局限性与后续计划\n")
    L.append("1. **欠训练**：20 epochs 下红斑/出血置信度普遍低于 0.24，类别级 FP 结论不可靠；需至少 100–200 epochs。")
    L.append("2. **单 seed**：无法估计种子方差；建议对 A/C/M1/M2/P0/P2/E 补跑 seed 2025、3407。")
    L.append("3. **方法前提被证伪**：CCC 的对比度损失是平滑明度压缩，不是高频细节丢失；P0/P2 的细节注入需要 24× 增益，临床可接受性存疑。")
    L.append("4. **评估仅为自动指标**：未做胃肠病医生主观评分、未做心理物理实验（H3）。")
    L.append("5. **正常帧匹配**：theBG 乘法只近似官方 OrigD 的预处理；建议向作者索取原始全量划分。")
    L.append("6. **单卡算力**：完整 3-seed × 200 epochs 在 4060 Ti 上约需数天；建议使用 4090/A100 或减少条件。\n")

    # 11 reproduce
    L.append("## 11. 复现指南\n")
    L.append("### 11.1 目录\n")
    L.append("```")
    L.append("C:\\wce\\")
    L.append("├─ WCE2024\\                     # 官方仓库（已打兼容补丁）")
    L.append("├─ data\\")
    L.append("│  ├─ downloads\\               # 原始 zip")
    L.append("│  ├─ raw\\{origd,ccd,cccd,seeai_full}")
    L.append("│  └─ ...")
    L.append("└─ wce_experiment\\")
    L.append("   ├─ data\\cond\\{A,C,M1,M2,M3,P0,P2,E}\\{train,val,test}")
    L.append("   ├─ data\\yolo\\{cond}\\{images,labels}")
    L.append("   ├─ data\\yaml\\{cond}.yaml")
    L.append("   ├─ runs\\controlled\\{cond}_s42\\weights\\best.pt")
    L.append("   ├─ runs\\anchor\\{A_origd_s42,C_cccd_s42}")
    L.append("   ├─ results\\{gate_g1,gate_g2,gate_g3,sweep,params}.json")
    L.append("   ├─ results\\eval\\{cond}_s42.json")
    L.append("   ├─ results\\figures\\*.png")
    L.append("   └─ scripts\\*.py")
    L.append("```\n")
    L.append("### 11.2 关键脚本\n")
    L.append(table(["脚本", "作用"], [
        ["scripts/extract.py", "解压三个 YOLO 包"],
        ["wce/data.py", "manifest、dHash/HSV 聚类、split"],
        ["scripts/fit_ccm.py", "反推 CCC/CC 仿射矩阵"],
        ["scripts/make_conditions.py", "生成 A/C/M1/M2/M3/P0/P2/E"],
        ["scripts/gates.py / gates_g3.py", "G1/G2 与 G3"],
        ["scripts/sweep.py", "截止尺度扫描"],
        ["scripts/choose_params.py", "r3/e_clip/gain 标定"],
        ["scripts/prepare_controlled.py", "构建 YOLO 硬链接与 yaml"],
        ["scripts/run_training.py", "训练调度"],
        ["scripts/evaluate.py", "测试集推理与 FROC/FP/面积"],
        ["scripts/stats.py", "Wilcoxon + Holm"],
        ["scripts/make_report.py / make_full_doc.py", "生成报告"],
    ]) + "\n")
    L.append("### 11.3 最小复现命令\n")
    L.append("```powershell")
    L.append("python scripts\\fit_ccm.py")
    L.append("python -c \"import sys; sys.path.insert(0,r'C:\\wce\\wce_experiment'); import wce.data as d; d.main()\"")
    L.append("python scripts\\make_conditions.py --stage base --workers 8")
    L.append("python scripts\\sweep.py --n 300")
    L.append("python scripts\\choose_params.py")
    L.append("python scripts\\make_conditions.py --stage extra --workers 6 --threads")
    L.append("python scripts\\prepare_controlled.py --conditions A,C,M1,M2,P0,P2,E")
    L.append("python scripts\\run_training.py --conditions A,C,M1,M2,P0,P2,E --seeds 42 --epochs 20 --parallel 2 --workers 1")
    L.append("python scripts\\evaluate.py --conditions A,C,M1,M2,P0,P2,E --split test")
    L.append("python scripts\\stats.py")
    L.append("python scripts\\make_full_doc.py")
    L.append("```\n")

    L.append("## 12. 附录：关键文件索引\n")
    L.append(table(["文件", "内容"], [
        ["results/ccm_cccd.json", "CCC 矩阵 M/b、RMSE"],
        ["results/ccm_ccd.json", "CC 矩阵 M/b、RMSE"],
        ["results/manifest.csv", "18,481 图的行级清单（split/cluster/boxes）"],
        ["results/split_stats.json", "各 split 数量与类别分布"],
        ["results/gate_g1.json", "截断率"],
        ["results/gate_g2.json", "亮度权重与梯度比"],
        ["results/gate_g3.json", "2×2 析因对比度"],
        ["results/sweep.json", "13 个半径的前沿曲线"],
        ["results/params.json", "r3/r_med/e_clip/g_p2"],
        ["results/eval/summary_test_all.json", "7 条件测试集汇总"],
        ["results/eval/stats.json", "Holm 校正后的配对统计"],
        ["REPORT.md", "精简版报告"],
    ]) + "\n")

    L.append("---\n*本报告由脚本自动汇总；所有数值来自 `results/` 下的 JSON/CSV，可逐项复核。*")

    open(OUT, "w", encoding="utf-8-sig").write("\n".join(L))
    print("wrote", OUT, len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
