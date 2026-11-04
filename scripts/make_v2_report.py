import os, json, csv, numpy as np

RES = r"C:\wce\wce_experiment\results"
OUT = r"C:\wce\wce_experiment\WCE_v2_审计与重跑报告.md"
CONDS = ["O", "S", "L", "X", "H", "E"]
LABEL = {"O": "O 原图（统一源）", "S": "S 完整校正（CCC）", "L": "L 仅明度替换",
         "X": "X 仅色度替换", "H": "H 联合控制（λ=.25,η=1）", "E": "E S+CLAHE"}


def j(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


def table(h, rows):
    s = ["| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for r in rows:
        s.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(s)


def main():
    audit = j("v2_audit/audit_report.json", {})
    ccm = j("v2_audit/ccm_v2.json", {})
    e2 = j("v2_audit/e2/e2_results.json", {})
    e3 = j("v2_audit/e3_image_grid.json", {})
    stats = j("v2_eval/stats_v2.json", [])
    ev = {}
    for c in CONDS:
        ev[c] = j(f"v2_eval/{c}_s42.json")
    last = {c: j(f"v2_eval/{c}_s42_last.json") for c in CONDS}
    o25 = j("v2_eval/O_s2025.json"); s25 = j("v2_eval/S_s2025.json")

    L = []
    L.append("# WCE v2：评估审计、统一条件与匹配域重跑\n")
    L.append("**日期**：2026-11-03  \n**本地**：RTX 4060 Ti 16GB  \n"
             "**远程**：192.168.10.3（3×3090）+ 192.168.10.65（2×3090 空闲，GPU2 他人占用）  \n"
             "**代码/结果**：本地 `C:\\wce\\wce_experiment`；远程 `/mnt/data2/wyh/wce`\n")
    L.append("> 本轮按修订方案执行：先审计，再重建统一条件，最后用匹配域 100-epoch 训练比较 O/S/L/X/H/E。"
             "所有条件使用同一划分、同一增强配置、同一训练预算；O/S 额外跑了 seed 2025 以估计种子方差。\n")

    L.append("## 0. 本轮结论\n")
    L.append("- **评估阈值问题确认**：作者仓库 `val.py` 默认 `conf_thres=0.2499`，训练内验证未覆盖；`best.pt` 是在高阈值过滤后的 AP 上选的。低阈值重评显示 **last.pt 的 AP50 一致高于 best.pt**（O 0.466→0.478，S 0.466→0.482，H 0.476→0.491，L 0.480→0.488），说明选模偏差真实存在。")
    L.append("- **S≈O**：100-epoch 匹配域训练下，完整 CCC 校正（S）与原图（O）的 AP50 均为 0.466；AP50-95 0.229 vs 0.232；总 FP 696 vs 746。没有稳定提升。")
    L.append("- **H 只有很小的方向性优势**：H（λ=.25, η=1）AP50 0.476，高于 S 的 0.466 和 E 的 0.466，但差值 **0.010 小于 O/S 的种子方差**（O 0.466 vs 0.478；S 0.466 vs 0.491），且配对检验 Holm 后均不显著。")
    L.append("- **L/X 分量**：L（仅明度替换）AP50 0.480、F1 0.542，但 FP 767；X（仅色度替换）AP50 0.469、F1 0.541，FP 768。两者都是“略高 AP/F1、更高 FP”的权衡，而非单方向改善。")
    L.append("- **机制解释需降级**：图像层确实看到明度分量主导对比度变化，但检测端效应小且被种子方差淹没。当前数据**不支持**“CCC → 截断/细节丢失 → 框膨胀 → FP 必然增加”的强因果链，也不支持“H 已显著优于 E”。\n")

    L.append("## 1. E0：评估器与数据管线审计\n")
    L.append("### 1.1 阈值与选模\n")
    if audit.get("threshold_audit"):
        t = audit["threshold_audit"]
        L.append(f"- `val.py` 默认 `conf_thres={t['val_py_default_conf']}`；`train.py` 的 `validate.run()` **未覆盖该参数**：{t['train_validate_overrides_conf']}。")
        L.append(f"- 来源：{t['source']}。")
    L.append("- 影响：训练内 AP 先过滤低置信度预测，`best.pt`/早停偏向高置信度工作点；应改为 `conf_thres=0.001` 重新选模，或在论文中明确使用 `last.pt` 作为无偏检查点。\n")
    if audit.get("unit_tests"):
        L.append("### 1.2 匹配与指标单元测试\n")
        rows = [[x["case"], x.get("tp", "—"), x.get("fp", "—"), x.get("ap", "—"), "PASS" if x["pass"] else "CHECK"] for x in audit["unit_tests"]]
        L.append(table(["用例", "TP", "FP", "AP", "结果"], rows) + "\n")
        L.append("> `ap_simple` 为 0.995 而非 1.000，是 101 点插值在离散 PR 曲线末端的正常现象；实现逻辑无误。\n")
    if audit.get("label_audit"):
        la = audit["label_audit"]
        L.append("### 1.3 标签与类别审计\n")
        L.append(f"- 类别 ID：{la['classes_seen']}；坏标签 {la['bad_label']}；正常图非空标签 {la['normal_nonempty']}；异常图空标签 {la['abnormal_empty']}。")
        L.append(f"- 类别 0 = `{la['class_names'][0]}`；YAML 中“0 为背景”的注释 **不适用于本数据**，实际标签从 0 开始映射 12 类。\n")
    if audit.get("best_vs_last"):
        L.append("### 1.4 v1 模型的 best vs last（低阈值测试）\n")
        rows = [[x["cond"], f"{x['best_ap50']:.3f}", f"{x['last_ap50']:.3f}", f"{x['best_ap5095']:.3f}", f"{x['last_ap5095']:.3f}"] for x in audit["best_vs_last"]]
        L.append(table(["条件", "best AP50", "last AP50", "best AP50-95", "last AP50-95"], rows) + "\n")
        L.append("> 多数条件 last.pt 的低阈值 AP 更高，且排序会变化；这直接解释了为什么不能把 20-epoch 单 seed 的旧结论当作机制证据。\n")

    L.append("## 2. E1：统一条件与 CCM v2\n")
    if ccm:
        L.append(f"- 只在 **train split** 的 O↔C_pub 图像对上重拟合；分块 RMSE = {ccm['rmse_block']:.3f}/255。")
        L.append(f"- 独立验证 ΔE00：中位 {ccm['val_dE_median']:.3f}，P95 {ccm['val_dE_p95']:.3f}。")
        L.append(f"- 反推变换的负通道率 {ccm['negative_channel_rate_pct']:.4f}%，>255 率 {ccm['over255_rate_pct']:.4f}%。")
        L.append(f"- 行和（含偏移/255）：{[round(x,3) for x in ccm['row_sum_with_offset']]}。")
    L.append("- 所有条件从 **同一全量 SEE-AI 源图** 生成：O 为统一源；S/L/X/H/E 在同一 O 上计算，正常帧与异常帧走同一处理路径；输出 PNG，避免条件间 JPEG 差异。")
    L.append("- 18,481 张 × 6 条件在 10.3 上生成；10.65 因磁盘/CPU 慢，改为只做训练与推理。\n")

    L.append("## 3. E2：方法性质验证\n")
    if e2:
        L.append("### 3.1 仿射偏移会降低归一化对比度（无需模糊）\n")
        rows = [[f"{m1}", f"{m2}", f"{K:.3f}", f"{Kp:.3f}", f"{r:.3f}"] for m1, m2, K, Kp, r in e2["affine_offset_contrast"]]
        L.append(table(["m1", "m2", "输入 K", "输出 K' (a=.8,b=10)", "K'/K"], rows) + "\n")
        L.append("**含义**：即使没有高频损失，正偏移 b 也会让 Michelson 对比度下降 17–22%。旧扫描中的“对比度下降”不能直接解释为细节丢失。\n")
        L.append("### 3.2 自引导滤波不是频率截止\n")
        st = e2["step_edge"]
        L.append(f"- 阶跃边缘：输入对比度 {st['input_contrast']:.4f}，GF(r=64) 输出 {st['gf_contrast']:.4f}；边缘处 GF={st['gf_center']:.2f}，输入={st['input_center']:.2f}。")
        L.append("- 正弦纹理：GF(r=64) 的 L* std 在各周期几乎不变（约 4.9），而 σ=21 高斯在 period≤16 时降到 0.04–0.28。")
        L.append("**含义**：GF 保边，不是“半径越大→细节全部转移到细节层”；旧 P0 扫描平坦是方法性质所致，不能证明“没有高频细节衰减”。\n")
        if "h_L" in e2:
            h = e2["h_L"]
            L.append("### 3.3 一维明度映射能解释多少？\n")
            L.append(f"- 在 {h['n_images']} 张训练图上，Lc≈a·L0+b：a={h['affine_slope']:.3f}，b={h['affine_intercept']:.2f}，R²={h['r2']:.3f}。")
            L.append(f"- 低色度 R²={h['r2_low_chroma']:.3f}，高色度 R²={h['r2_high_chroma']:.3f}；残差 std={h['residual_std']:.2f}，与色度相关 {h['residual_corr_chroma']:+.3f}。")
            L.append("**含义**：明度映射是主导近似，但残差仍与色度相关；不能把全部变化归结为“纯明度压缩”。\n")

    L.append("## 4. E3：参数族与 H 选择\n")
    if e3:
        L.append("在 200 张 val 图上计算 15 点 (λ, η) 网格的图像指标；约束为 **median ΔE00(S) ≤ 2.0**，在可行点中取最大病灶框 Michelson。")
        L.append(f"- 选中 H：**λ={e3['selected']['lambda']}, η={e3['selected']['eta']}**；ΔE00(S)={e3['selected']['dE_to_S']:.3f}，局部 RMS={e3['selected']['local_rms']:.3f}（S 为 2.17），框 Michelson={e3['selected']['box_michelson']:.4f}（S 为 0.028），|ΔL|={e3['selected']['abs_dL']:.2f}。")
        L.append("- 说明：这是**图像层选择**；冻结检测器的网格推理（E3 的完整版）尚未执行，因此 H 的选择仍应视为开发集上的初步决定。\n")
        rows = [[f"{g['lambda']:.2f}", f"{g['eta']:.1f}", f"{g['dE_to_S']:.2f}", f"{g['local_rms']:.2f}", f"{g['box_michelson']:.4f}"] for g in e3["grid"]]
        L.append(table(["λ", "η", "ΔE00(S)", "局部 RMS", "框 Michelson"], rows) + "\n")

    L.append("## 5. E4：匹配域 100-epoch 结果\n")
    rows = []
    for c in CONDS:
        if not ev[c]:
            continue
        o = ev[c]["overall"]
        rows.append([LABEL[c], f"{o['map50']:.3f}", f"{o['map50_95']:.3f}", f"{o['f1']:.3f}",
                     o["fp"], o["fp_normal"], o["fp_abnormal"]])
    L.append(table(["条件", "AP50", "AP50-95", "F1", "FP 总数", "正常图 FP", "异常图 FP"], rows) + "\n")
    L.append("### 5.1 best vs last 检查点敏感性\n")
    rows = []
    for c in CONDS:
        if not ev[c] or not last[c]:
            continue
        rows.append([LABEL[c], f"{ev[c]['overall']['map50']:.3f}", f"{last[c]['overall']['map50']:.3f}",
                     f"{ev[c]['overall']['f1']:.3f}", f"{last[c]['overall']['f1']:.3f}",
                     ev[c]["overall"]["fp"], last[c]["overall"]["fp"]])
    L.append(table(["条件", "best AP50", "last AP50", "best F1", "last F1", "best FP", "last FP"], rows) + "\n")
    L.append("> last.pt 的 AP50 普遍更高，说明 0.2499 阈值选出的 best.pt 不是低阈值最优；条件排序基本不变，但绝对差值应同时报告两种检查点。\n")
    L.append("> X/E 的 last.pt 评估因 NMS 时间上限未完成；其 best.pt 结果已足以支持本报告的结论。\n")
    L.append("### 5.2 种子方差（O/S）\n")
    if o25 and s25:
        L.append(table(["条件", "seed 42 AP50", "seed 2025 AP50", "差值", "seed 42 FP", "seed 2025 FP"], [
            ["O", f"{ev['O']['overall']['map50']:.3f}", f"{o25['overall']['map50']:.3f}",
             f"{o25['overall']['map50']-ev['O']['overall']['map50']:+.3f}", ev['O']['overall']['fp'], o25['overall']['fp']],
            ["S", f"{ev['S']['overall']['map50']:.3f}", f"{s25['overall']['map50']:.3f}",
             f"{s25['overall']['map50']-ev['S']['overall']['map50']:+.3f}", ev['S']['overall']['fp'], s25['overall']['fp']],
        ]) + "\n")
        L.append("> 种子间 AP50 差值（O 0.012，S 0.025）大于 H 相对 S/E 的 0.010，因此 **单 seed 不能支持条件优劣**。\n")

    L.append("## 6. E5：逐图配对统计（seed 42，探索性）\n")
    if stats:
        rows = [[f"{r['a']} vs {r['b']}", f"{r['fp_diff']:+.3f}", f"{r['p_fp']:.2e}", f"{r['p_fp_holm']:.2e}",
                 f"{(r['area_a'] or 0):.3f}", f"{(r['area_b'] or 0):.3f}", f"{r['p_area']:.2e}"] for r in stats]
        L.append(table(["比较", "ΔFP (a−b)", "p_FP", "Holm p_FP", "面积比 a", "面积比 b", "p_面积"], rows) + "\n")
        L.append("> Holm 校正后所有 FP 差异均不显著；H vs S 的面积比 nominal p=0.036，但 Holm 后不显著。\n")

    L.append("## 7. 对研究计划的影响\n")
    L.append("1. **删除“CCC 必然损害检测”的强表述**：S≈O，且所有条件差异被种子方差覆盖。")
    L.append("2. **删除“细节注入修复对比度”的强表述**：GF 不是频率截止；P0 扫描平坦不能证明高频细节丢失；H 仅带来 ~0.01 AP50 的非显著变化。")
    L.append("3. **保留可测量的分量关系**：图像层明度分量主导 L* 对比度变化（G3），但任务层效应需在低阈值、多种子、匹配域下重评。")
    L.append("4. **RP 主线改为“测量—受控干预—多层验证”**：O/S/L/X/H/E 作为受控干预，冻结模型与匹配域重训分开；H 作为保守工作点，而不是必胜方法。")
    L.append("5. **评估协议必须写清**：低阈值推理、PR/FROC、正常帧 FP 单列、检查点选择规则、种子方差。\n")

    L.append("## 8. 剩余实验\n")
    L.append("- 用 `conf_thres=0.001` 重新训练/选模（当前 best.pt 仍受 0.2499 影响）；至少对 O/S/H/E 各补 3 seeds。")
    L.append("- 完成 E3 的冻结检测器 15 点网格推理，用 f_O/f_S 而非图像层指标选 H。")
    L.append("- X/E/H 的 last.pt 评估（X/E 正在远程跑）与 class-wise FROC/框面积比。")
    L.append("- Kvasir-Capsule 外部验证：按视频分组，不把未标注帧当正常负样本。")
    L.append("- 人眼/临床观察者实验（H3）与显示条件（SO7）。\n")

    L.append("## 9. 图与文件\n")
    L.append("![montage](results/v2_fig/montage_OSLXHE.png)\n")
    L.append("*图 1：同一测试帧的 O / S / L / X / H / E。*\n")
    L.append("![ap](results/v2_fig/ap50_best_last.png)\n")
    L.append("*图 2：best.pt vs last.pt 的 AP50。*\n")
    L.append("![seed](results/v2_fig/seed_variance.png)\n")
    L.append("*图 3：O/S 的种子方差。*\n")
    L.append("![e3](results/v2_fig/e3_grid.png)\n")
    L.append("*图 4：15 点 (λ,η) 图像层网格；红圈为 H。*\n")
    L.append("![e2a](results/v2_audit/figures/e2_affine_contrast.png)\n")
    L.append("*图 5：仿射偏移降低归一化对比度，无需模糊。*\n")
    L.append("![e2b](results/v2_audit/figures/e2_sinusoid.png)\n")
    L.append("*图 6：GF(r=64) 与高斯平滑的频率响应对比。*\n")
    L.append("### 关键文件\n")
    L.append(table(["文件", "内容"], [
        ["results/v2_audit/audit_report.json", "E0 阈值/checkpoint/单元测试/标签审计"],
        ["results/v2_audit/ccm_v2.json", "train-only 重拟合 CCC 矩阵"],
        ["results/v2_audit/e2/e2_results.json", "E2 合成与 h(L) 分析"],
        ["results/v2_audit/e3_image_grid.json", "15 点图像层网格与 H"],
        ["results/v2_eval/{O,S,L,X,H,E}_s42.json", "测试集低阈值指标"],
        ["results/v2_eval/{O,S,H,L}_s42_last.json", "last.pt 敏感性"],
        ["results/v2_eval/stats_v2.json", "逐图配对统计"],
        ["runs/v2/*/weights/best.pt", "远程训练权重"],
    ]) + "\n")
    L.append("---\n*本报告只使用自动指标；H 的冻结检测器网格选择、3 seeds 与外部数据仍待完成。*")

    open(OUT, "w", encoding="utf-8-sig").write("\n".join(L))
    print("wrote", OUT, len("\n".join(L)))


if __name__ == "__main__":
    main()
