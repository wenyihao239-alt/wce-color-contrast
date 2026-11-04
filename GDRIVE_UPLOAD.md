# Google Drive 上传说明

准备日期：2026-11-04

## 最终体积

| 部分 | 体积 |
|---|---:|
| V1 条件图（8 条件） | 7.4 GB |
| V2 条件图（6 条件） | 33.9 GB |
| 模型权重 | 0.46 GB |
| 标签/JSON/文档 | < 0.1 GB |
| **合计** | **~42.5 GB** |

## 推荐上传顺序

1. models/（先传，最小）
2. metadata/（标签、清单、JSON、报告）
3. datasets/V1/（可选，历史探索）
4. datasets/V2/（主分析，建议优先）

## 空间不足时的最小集

models/ + metadata/ + datasets/V2/{O,S,H,E} ≈ 20 GB。  
其余条件可用 GitHub 里的脚本从 O/S 重新生成。

## 数据对应关系

- V1：datasets/V1/{A,C,M1,M2,M3,P0,P2,E}/{train,val,test}/，JPEG 图像；标签在 metadata/labels/{split}/，按同名 txt 配对。
- V2：datasets/V2/{O,S,L,X,H,E}/{train,val,test}/，PNG 图像；O/S 的标签用 metadata/labels/，L/X/H/E 目录下也有 labels/ 副本。
- 所有图像 576×576；类别 ID 0–11 见 metadata/manifests/ 或报告。

## 校验

- 模型与小型文件：checksums_models.sha256
- 各条件/各 split 的图像数与字节数：dataset_filecounts.csv
- 上传后对照这两个文件即可；不要求对 20 万小图逐个算哈希。

## 许可

SEE-AI / kyucapsule 与 Agossou 校正数据集均为 CC BY 4.0；派生条件图沿用 CC BY 4.0。  
使用时请引用 SEE-AI（Yokote et al. 2024）、WCE2024（Agossou et al. 2025）和 Watine et al. 2023。
