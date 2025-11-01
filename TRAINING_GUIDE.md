# DiffMLP 训练指南

## 快速开始

### 训练模型

```bash
cd /home/liuhao/diff-mlp

# 训练 (GPU 0)
bash experiment-diff.sh configs/fb15k-237-diffmlp.sh --train 0

# 训练 (GPU 1)
bash experiment-diff.sh configs/fb15k-237-diffmlp.sh --train 1
```

### 推理评估

```bash
# 测试集评估 (自动显示 MPS 指标)
bash experiment-diff.sh configs/fb15k-237-diffmlp.sh --inference 0
```

---

## 配置说明

配置文件: [`configs/fb15k-237-diffmlp.sh`](configs/fb15k-237-diffmlp.sh)

### 关键参数

**扩散参数** (已优化):
```bash
diff_T_max=600        # 最大扩散步数
diff_gamma=2.0        # 自适应步数敏感度
diff_delta=0.90       # 早停阈值 (↑ 更充分去噪)
diff_eta=0.01         # 先验约束步长 (↑ 更强约束)
diff_vartheta=0.2     # 语义一致性损失权重 (↑ 更强语义)
diff_num_layers=4     # GAT 去噪层数
```

**训练参数**:
```bash
num_epochs=100        # 训练轮数
learning_rate=0.001   # 学习率
batch_size=16         # 批大小
num_rollouts=20       # 每个查询的 rollout 数
num_rollout_steps=3   # 推理跳数
```

---

## 评估指标

### 链接预测指标
- **Hits@1, Hits@3, Hits@5, Hits@10**: 前 K 命中率
- **MRR**: 平均倒数排名

### 路径质量指标
- **MPS**: Mean Path Spuriousness (路径虚假度)
  - 基于路径置信度计算
  - **越低越好** (表示路径越可靠)
  - 当前实现: `MPS = 1 - IMPS`

---

## 优化建议

### 提升 Hits@10
1. 增加 `beam_size` (默认 128)
2. 增加 `num_rollouts` (默认 20)
3. 调整 `learning_rate`

### 降低 MPS (提升路径质量)
1. 增加 `diff_vartheta` (语义一致性权重)
2. 增加 `diff_eta` (先验约束强度)
3. 增加 `diff_num_layers` (去噪层数)
4. 提高 `diff_delta` (早停阈值)

### 平衡性能与速度
- 减少 `diff_T_max` 加快训练
- 减少 `num_rollouts` 降低内存
- 减少 `diff_num_layers` 提升速度

---

## 常见问题

### Q: 训练很慢怎么办？
A: 
- 减少 `num_rollouts` 到 10-15
- 减少 `diff_num_layers` 到 2-3
- 使用更大的 `batch_size`

### Q: MPS 指标没有显示？
A: 推理时会自动启用 `--save_beam_search_paths`，确保使用 `--inference` 模式

### Q: 如何调整超参数？
A: 直接修改 `configs/fb15k-237-diffmlp.sh` 中的参数值

---

## 预期结果

| 指标 | 预期值 | 说明 |
|------|--------|------|
| Hits@10 | 0.71-0.75 | 前10命中率 |
| MRR | 0.39-0.42 | 平均倒数排名 |
| MPS | 0.15-0.20 | 路径虚假度 (越低越好) |

---

## 文件结构

```
diff-mlp/
├── configs/
│   └── fb15k-237-diffmlp.sh          # 配置文件
├── src/
│   ├── rl/graph_search/
│   │   ├── diff_pn.py                # DiffMLP 策略网络
│   │   └── diff_pg.py                # DiffMLP 训练器
│   ├── eval.py                       # 评估 (含 MPS)
│   └── experiments.py                # 主程序
├── experiment-diff.sh                # 训练脚本
└── TRAINING_GUIDE.md                 # 本文档
```

---

## 进阶: 论文要求 vs 当前实现

### 已实现 ✅
- DiffMLP 核心扩散机制
- 条件编码器 (ConditionalEncoder)
- GAT 去噪器 (GATDenoiserLayer)
- 语义一致性损失 L_g
- 自适应扩散步数
- 早停机制
- 先验约束

### 待优化 ⚠️
1. **MPS 计算方法**
   - 当前: 基于置信度 (`MPS = 1 - IMPS`)
   - 论文: 路径替换方法 (`MPS = PS(ρ, r_q) / PS(ρ)`)

2. **路径奖励**
   - 当前: 仅使用实体奖励 R_e
   - 论文: R_K = R_e + R_p (需要路径奖励)

### 改进方向
- 实现正确的路径替换 MPS
- 添加路径奖励 R_p (基于 LSTM 相似度)
- 集成预训练 embedding 模型

---

## 联系与支持

如有问题，请检查:
1. 数据是否正确加载 (`data/FB15K237/`)
2. GPU 是否可用
3. 依赖是否安装完整

训练日志保存在工作目录，可用于调试。
