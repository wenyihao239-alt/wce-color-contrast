import os, json, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = r"C:\wce\wce_experiment\results"
FIG = os.path.join(RES, "v2_fig"); os.makedirs(FIG, exist_ok=True)
OUT = r"C:\wce\wce_experiment\WCE_v2.1_评估收尾报告.md"
CONDS = ["O", "S", "L", "X", "H", "E"]
LABEL = {"O": "O 原图", "S": "S CCC 渲染", "L": "L 仅明度", "X": "X 仅色度",
         "H": "H 联合控制", "E": "E S+CLAHE"}
EPOCHS = {"O": 92, "S": 90, "L": 99, "X": 93, "H": 97, "E": 93}


def load(path):
    p = os.path.join(RES, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def table(h, rows):
    s = ["| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for r in rows:
        s.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(s)


def main():
    audit = load("v2_audit/audit_report.json") or {}
    ccm = load("v2_audit/ccm_v2.json") or {}
    e2 = load("v2_audit/e2/e2_results.json") or {}
    e3 = load("v2_audit/e3_image_grid.json") or {}
    stats = load("v2_eval/stats_v2.json") or []
    ev = {c: load(f"v2_eval/{c}_s42.json") for c in CONDS}
    last = {c: load(f"v2_eval/{c}_s42_last.json") for c in CONDS}
    o25 = load("v2_eval/O_s2025_last.json"); s25 = load("v2_eval/S_s2025_last.json")
    cross = {}
    for m in ["O", "S"]:
        for i in CONDS:
            d = load(f"v2_cross/cross{m}_{i}_s42_last.json")
            if d:
                cross[(m, i)] = d

    # cross heatmap
    if cross:
        mat = np.array([[cross[(m, i)]["overall"]["map50"] for i in CONDS] for m in ["O", "S"]])
        plt.figure(figsize=(6.4, 2.4))
        im = plt.imshow(mat, cmap="viridis", aspect="auto")
        plt.colorbar(im, label="AP50")
        plt.xticks(range(len(CONDS)), CONDS); plt.yticks([0, 1], ["f_O (O-trained)", "f_S (S-trained)"])
        for r in range(2):
            for c in range(len(CONDS)):
                plt.text(c, r, f"{mat[r,c]:.3f}", ha="center", va="center",
                         color="white" if mat[r, c] < 0.35 else "black", fontsize=8)
        plt.title("Frozen-model cross-input AP50 (val)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "cross_heatmap.png"), dpi=180); plt.close()

    L = []
    L.append("# WCE V2.1：评估收尾、固定检查点重评与冻结模型交叉推理\n")
    L.append("**日期**：2026-11-04  \n**本地**：RTX 4060 Ti 16GB；**远程**：192.168.10.3（3×3090）+ 192.168.10.65（2×3090）  \n"
             "**范围**：这是申请前的探索性技术收尾，不是验证性研究；H 参数已冻结，不再根据测试结果改选。\n")
    L.append("## 0. 结论\n")
    L.append("- **未建立稳定的条件排序**。在固定的 final-epoch 检查点下，O/S/L/X/H/E 的 AP50 为 0.478–0.491；H 最高（0.491），但相对 S（0.482）仅 +0.008，与 S 的 run-to-run 差异（0.482 / 0.491）同量级。")
    L.append("- **不能称为“等价”**。Holm 检验主要针对 FP 计数，不能用于判定 AP 等价；正确表述是“当前协议下未观察到稳健的性能差异”。")
    L.append("- **检查点敏感但原因未完全分离**：`best.pt` 与 `last.pt` 的测试 AP50 差异明显；由于选模 fitness 是 0.1·AP50 + 0.9·AP50-95、且验证/测试不同，不能把差异单独归因于 0.2499 阈值。`last.pt` 只应称为“固定最终轮次检查点”，不是“无偏”。")
    L.append("- **NMS 完整性已修复**：把 `time_limit` 置为无穷大后，8 个主分析的 `incomplete_batches` 均为 0。max_det 触顶仍频繁，但对 O 的 300 vs 1000 敏感性测试显示 AP50 仅变化 0.0007、FP 不变。")
    L.append("- **输入域偏移远大于条件内差异**：f_O(O)=0.491 → f_O(S)=0.219；f_S(S)=0.484 → f_S(O)=0.227；f_S(H)=0.458。冻结模型跨域推理出现灾难性下降，说明“匹配域训练”是必要控制，而不是可选细节。")
    L.append("- **H 仍是保守工作点，不是已验证的优胜方法**。\n")

    L.append("## 1. E0：评估与选模审计\n")
    L.append(f"- `val.py` 默认 `conf_thres=0.2499`，训练内 `validate.run()` 未覆盖；选模 fitness 为 0.1·AP50 + 0.9·AP50-95，且在验证集上计算。")
    L.append("- v1 的 best/last 低阈值重评显示：A 0.268→0.310、C 0.251→0.292、M1 0.291→0.276、M2 0.290→0.312、P0 0.282→0.282、P2 0.283→0.283、E 0.276→0.305。")
    L.append("- **NMS 修复**：仓库在 `time_limit = 0.5 + 0.05·batch` 超时后 `break`，未处理图像保留空预测。已改为 `1e9`；重评后所有主分析 `incomplete_batches=0`。")
    L.append("- **max_det 敏感性**：O last 在 max_det=300 时 AP50=0.4783、FP=894；max_det=1000 时 AP50=0.4776、FP=894。触顶频繁但指标稳定，主表保留 300。")
    L.append("- 单元测试：exact_single / wrong_class / duplicate_preds / gt_no_pred / pred_no_gt / empty 全部通过；AP 单例为 0.995 是 101 点插值末端效应，需用固定参考实现对照，不能笼统解释。")
    L.append("- 标签审计：类别 0–11 均出现，坏标签 0，正常图非空标签 0，异常图空标签 0；类别 0 = angiodysplasia。\n")

    L.append("## 2. E1：统一条件与 CCM v2\n")
    if ccm:
        L.append(f"- 仅在 train split 的 O↔C_pub 对上重拟合；块均值 RMSE={ccm['rmse_block']:.3f}/255；val 中位 ΔE00={ccm['val_dE_median']:.3f}，P95={ccm['val_dE_p95']:.3f}。")
        L.append(f"- 负通道率 {ccm['negative_channel_rate_pct']:.4f}%，>255 率 {ccm['over255_rate_pct']:.4f}%。")
    L.append("- 所有条件由同一全量 SEE-AI 源图生成（异常/正常同路径，PNG），S 仅为“由图像对拟合的 CCC 渲染近似”，不是经物理测量验证的 SB3 标定。\n")

    L.append("## 3. E2：方法性质\n")
    if e2:
        L.append("- 仿射偏移 (a=.8,b=10) 使归一化 Michelson 对比度降至输入的 0.78–0.83，**无需任何模糊**。")
        st = e2["step_edge"]
        L.append(f"- 自引导 GF(r=64) 对阶跃边缘的对比度 {st['input_contrast']:.4f}→{st['gf_contrast']:.4f}，几乎不变；正弦纹理 std 在各周期约 4.9，而 σ=21 高斯在短周期降至 0.04–0.28。**GF 不是频率截止**。")
        h = e2["h_L"]
        L.append(f"- Lc≈a·L0+b：a={h['affine_slope']:.3f}，R²={h['r2']:.3f}；低色度 R²={h['r2_low_chroma']:.3f}，高色度 R²={h['r2_high_chroma']:.3f}；残差 std={h['residual_std']:.2f}，与色度相关 {h['residual_corr_chroma']:+.3f}。一维明度映射是主导近似，但不完整。\n")

    L.append("## 4. E3：H 的固定\n")
    if e3:
        L.append(f"H=(λ={e3['selected']['lambda']}, η={e3['selected']['eta']})，由 200 张 val 图的 15 点网格选出；约束 median ΔE00(S)≤2，最大化病灶框 Michelson。")
        L.append(f"选中点：ΔE00(S)={e3['selected']['dE_to_S']:.3f}，局部 RMS={e3['selected']['local_rms']:.3f}（S=2.17），框 Michelson={e3['selected']['box_michelson']:.4f}（S=0.028）。")
    L.append("> ΔE00≤2 是探索性参考渲染偏差约束，不是临床可接受阈值或真实组织色差；H 不再根据检测器重新选择。\n")

    L.append("## 5. E4：主结果（final-epoch 检查点）\n")
    L.append("所有模型使用 `last.pt` 作为固定的最终轮次检查点；实际完成轮数写在表中。`best.pt` 仅作敏感性分析。\n")
    rows = []
    for c in CONDS:
        if not last[c]:
            continue
        o = last[c]["overall"]
        rows.append([LABEL[c], EPOCHS.get(c, "—"), f"{o['map50']:.3f}", f"{o['map50_95']:.3f}",
                     f"{o['f1']:.3f}", o["fp"], o["fp_normal"], o.get("n_maxdet"), o.get("incomplete_batches")])
    L.append(table(["条件", "实际 epochs", "AP50", "AP50-95", "F1@0.24", "FP 总数", "正常帧 FP", "max_det 触顶", "NMS 不完整批次"], rows) + "\n")
    L.append("### 5.1 best.pt 敏感性\n")
    rows = []
    for c in CONDS:
        if ev[c] and last[c]:
            rows.append([LABEL[c], f"{ev[c]['overall']['map50']:.3f}", f"{last[c]['overall']['map50']:.3f}",
                         f"{ev[c]['overall']['f1']:.3f}", f"{last[c]['overall']['f1']:.3f}"])
    L.append(table(["条件", "best AP50", "last AP50", "best F1", "last F1"], rows) + "\n")
    L.append("> 差异只能说明检查点敏感；不能单独归因于 0.2499，也不能把 last 称为无偏。\n")
    L.append("### 5.2 O/S 的 run-to-run 差异\n")
    if o25 and s25:
        L.append(table(["条件", "seed 42 AP50 (last)", "seed 2025 AP50 (last)", "差值", "FP 42", "FP 2025"], [
            ["O", f"{last['O']['overall']['map50']:.3f}", f"{o25['overall']['map50']:.3f}",
             f"{o25['overall']['map50']-last['O']['overall']['map50']:+.3f}", last['O']['overall']['fp'], o25['overall']['fp']],
            ["S", f"{last['S']['overall']['map50']:.3f}", f"{s25['overall']['map50']:.3f}",
             f"{s25['overall']['map50']-last['S']['overall']['map50']:+.3f}", last['S']['overall']['fp'], s25['overall']['fp']],
        ]) + "\n")
        L.append("> 这是两次运行的差异，不是方差；不能外推到 H/E，也不足以稳定估计训练随机性。\n")

    L.append("## 6. E5：冻结模型交叉推理（val）\n")
    if cross:
        rows = []
        for i in CONDS:
            rows.append([LABEL[i],
                         f"{cross[('O',i)]['overall']['map50']:.3f}" if ("O", i) in cross else "—",
                         f"{cross[('S',i)]['overall']['map50']:.3f}" if ("S", i) in cross else "—"])
        L.append(table(["输入条件", "f_O (O-trained)", "f_S (S-trained)"], rows) + "\n")
        L.append("**解读**：f_O 在 O 上 0.491，换到 S 掉到 0.219；f_S 在 S 上 0.484，换到 O 掉到 0.227。"
                 "输入域偏移造成的下降远大于同域条件间 0.01 量级的差异。H 更接近 S 域（f_S(H)=0.458，f_O(H)=0.255），"
                 "说明联合控制本质上仍是一个新的输入域；匹配域重训不可省略。\n")

    L.append("## 7. 逐图配对统计（seed 42，探索性）\n")
    if stats:
        rows = [[f"{r['a']} vs {r['b']}", f"{r['fp_diff']:+.3f}", f"{r['p_fp']:.2e}", f"{r['p_fp_holm']:.2e}",
                 f"{(r['area_a'] or 0):.3f}", f"{(r['area_b'] or 0):.3f}", f"{r['p_area']:.2e}"] for r in stats]
        L.append(table(["比较", "ΔFP (a−b)", "p_FP", "Holm p_FP", "面积比 a", "面积比 b", "p_面积"], rows) + "\n")
        L.append("> 这些 p 值只针对 FP/面积，不能用于判定 AP 等价；Holm 后均不显著。\n")

    L.append("## 8. 对申请版 RP 的写法约束\n")
    L.append("1. 写“under the current exploratory protocol, no robust performance ranking has been established”，不要写“S≈O”或“H 无效”。")
    L.append("2. 写“fixed final-epoch checkpoint”，不要写“unbiased”；best/last 差异仅作为检查点敏感性证据。")
    L.append("3. 写“run-to-run differences”，不要写“seed variance”后直接外推到其他条件。")
    L.append("4. H 的 ΔE00≤2 称为“exploratory reference-rendering constraint”，不是临床容差。")
    L.append("5. 先导工作只声称：统一数据/变换/评估管线、分量干预、匹配域训练、冻结模型交叉推理；不声称方法优越或机制已证。")
    L.append("6. NMS 完整性、max_det 敏感性、正常帧 FP 必须写进评估协议。\n")

    L.append("## 9. 剩余工作（不阻塞申请）\n")
    L.append("- H/E 的 seed 2025（2 次训练），再补 O/S/H/E 的 seed 3407（4 次训练）形成三种子核心比较。")
    L.append("- 用低阈值验证（`conf_thres=0.001`）重新选模或固定 final-epoch 规则，明确写入协议。")
    L.append("- Kvasir-Capsule 外部验证（按视频分组）与观察者实验；这些属于博士阶段，不是申请前置条件。\n")

    L.append("## 10. 图\n")
    L.append("![montage](results/v2_fig/montage_OSLXHE.png)\n")
    L.append("![cross](results/v2_fig/cross_heatmap.png)\n")
    L.append("![ap](results/v2_fig/ap50_best_last.png)\n")
    L.append("![e2a](results/v2_audit/figures/e2_affine_contrast.png)\n")
    L.append("![e2b](results/v2_audit/figures/e2_sinusoid.png)\n")

    open(OUT, "w", encoding="utf-8-sig").write("\n".join(L))
    print("wrote", OUT, len("\n".join(L)))


if __name__ == "__main__":
    main()
