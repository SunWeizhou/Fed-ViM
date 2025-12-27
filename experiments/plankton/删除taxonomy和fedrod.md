# 删除 Taxonomy 和 FedRoD 代码清理记录

本文档记录了从 pFL-FOOGD-Plankton 项目中删除 Taxonomy 和 FedRoD 相关代码的所有修改。

## 修改日期
2025-12-27

## 修改目标
1. 删除所有 Taxonomy-Aware Loss 相关代码
2. 删除 FedRoD 双头架构,简化为 FedAvg 单头架构
3. 保留 FedAvg + FOOGD 的核心功能

---

## 一、Taxonomy 代码删除

### 1.1 models.py
**删除内容:**
- 删除了 `from data_utils import build_taxonomy_matrix` 导入
- 删除了整个 `TaxonomyLoss` 类 (~70行代码)
  - 包含层级感知损失函数
  - 期望代价计算
  - 树正则化逻辑

**原因**: Taxonomy-Aware Loss 依赖生物分类学知识,对浮游生物数据集特定,不适用于通用场景

### 1.2 client.py
**删除内容:**
- 删除了 `from models import TaxonomyLoss` 导入
- 删除了 `__init__` 中的 `use_taxonomy` 和 `taxonomy_matrix` 参数
- 删除了 `tax_loss_fn` 初始化逻辑
- 删除了训练循环中的 taxonomy loss 计算分支
- 简化了 Head-G 的 loss 计算,只使用 CrossEntropy

**修改前:**
```python
if self.use_taxonomy and self.tax_loss_fn is not None:
    loss_g = self.tax_loss_fn(logits_g, targets)
else:
    loss_g = F.cross_entropy(logits_g, targets)
```

**修改后:**
```python
loss_g = F.cross_entropy(logits_g, targets)
```

### 1.3 eval_utils.py
**删除内容:**
- 删除了 `compute_hierarchical_error()` 函数
- 修改了 `evaluate_accuracy_metrics()` 函数:
  - 删除了 `taxonomy_matrix` 参数
  - 删除了层级错误(hier_error)和层级准确率(hier_acc)计算
  - 返回值从 `(id_acc, tail_acc, hier_error, hier_acc)` 改为 `(id_acc, tail_acc)`

### 1.4 data_utils.py
**删除内容:**
- 删除了整个 `build_taxonomy_matrix()` 函数 (~113行代码)
  - 包含生物分类学代价矩阵构建逻辑
  - 层级距离计算
  - 粗粒度和细粒度类别分组

### 1.5 train_federated.py
**删除内容:**
- 删除了 `from data_utils import create_federated_loaders, build_taxonomy_matrix` 中的 `build_taxonomy_matrix`
- 删除了 `create_clients()` 中的 `use_taxonomy` 和 `taxonomy_matrix` 参数
- 删除了分类学矩阵构建代码块 (~10行)
- 删除了 `--use_taxonomy` 命令行参数
- 删除了所有评估中的 `hier_error`, `hier_ratio` 相关变量
- 删除了 `evaluate_accuracy_metrics()` 调用中的 `taxonomy_matrix` 参数
- 删除了 `training_history['hierarchical_ratios']` 记录

---

## 二、FedRoD 代码删除

### 2.1 models.py
**修改内容:**

**类重命名:**
- `FedRoD_Model` → `FedAvg_Model`

**架构简化:**
```python
# 修改前 (FedRoD 双头)
class FedRoD_Model(nn.Module):
    def __init__(self, backbone, num_classes=54, hidden_dim=512):
        self.backbone = backbone
        self.head_g = nn.Sequential(...)  # 通用头
        self.head_p = nn.Sequential(...)  # 个性化头

    def forward(self, x):
        features = self.backbone(x)
        logits_g = self.head_g(features)
        logits_p = self.head_p(features)
        return logits_g, logits_p, features  # 返回3个值

# 修改后 (FedAvg 单头)
class FedAvg_Model(nn.Module):
    def __init__(self, backbone, num_classes=54, hidden_dim=512):
        self.backbone = backbone
        self.classifier = nn.Sequential(...)  # 单个分类头

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features  # 只返回2个值
```

**函数重命名:**
- `create_fedrod_model()` → `create_model()`

### 2.2 client.py
**删除内容:**
- 删除了 `__init__` 中的 `algorithm` 参数 (移除了 'fedavg'/'fedrod' 选择)
- 删除了 `self.algorithm` 属性
- 删除了 `evaluate_comprehensive()` 方法 (~100行)
  - 包含 Head-P 评估逻辑
  - ID/IN-C/Near/Far 全面评估

**修改内容:**

**训练循环简化:**
```python
# 修改前
logits_g, logits_p, features = self.model(data)

if self.algorithm == 'fedavg':
    classification_loss = loss_g
else:  # FedRoD
    classification_loss = loss_g + loss_p

# 修改后
logits, features = self.model(data)
classification_loss = F.cross_entropy(logits, targets)
```

**评估方法简化:**
```python
# 修改前
def evaluate(self, test_loader):
    logits_g, logits_p, _ = self.model(data)
    loss = (loss_g + loss_p) / 2
    return {'loss': ..., 'acc_g': ..., 'acc_p': ...}

# 修改后
def evaluate(self, test_loader):
    logits, _ = self.model(data)
    loss = F.cross_entropy(logits, targets)
    return {'loss': ..., 'acc': ...}
```

**参数获取简化:**
```python
# 修改前
def get_generic_parameters(self):
    for key, value in model_state.items():
        if 'head_p' not in key:  # 过滤掉 head_p
            params[f"model.{key}"] = value.clone()

# 修改后
def get_generic_parameters(self):
    for key, value in model_state.items():
        params[f"model.{key}"] = value.clone()
```

### 2.3 eval_utils.py
**删除内容:**
- 删除了所有 `use_head_g` 参数
- 删除了双头输出的判断逻辑

**修改前:**
```python
def evaluate_accuracy_metrics(model, dataloader, ..., use_head_g=False):
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        if use_head_g:
            logits = outputs[0]  # Head-G
        else:
            logits = outputs[1]  # Head-P
    else:
        logits = outputs
```

**修改后:**
```python
def evaluate_accuracy_metrics(model, dataloader, ...):
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        logits = outputs[0]  # 直接取第一个
    else:
        logits = outputs
```

**evaluate_id_performance 修改:**
```python
# 修改前
logits_g, logits_p, _ = model(data)
all_logits_g.extend(valid_logits_g.cpu().numpy())
all_logits_p.extend(logits_p[valid_mask].cpu().numpy())
metrics = {'logits_g': all_logits_g, 'logits_p': all_logits_p, ...}

# 修改后
logits, _ = model(data)
all_logits.extend(valid_logits.cpu().numpy())
metrics = {'logits': all_logits, ...}
```

### 2.4 train_federated.py
**删除内容:**
- 删除了 `--algorithm` 命令行参数定义
- 删除了 FedRoD 评估分支 (场景B, ~60行代码)
- 删除了 `client_test_loaders` 相关评估逻辑
- 删除了所有 Head-P 相关的指标记录

**修改前:**
```python
parser.add_argument('--algorithm', type=str, default='fedrod',
                   choices=['fedavg', 'fedrod'],
                   help='选择算法: fedavg 或 fedrod')

# 场景 A: FedAvg
if args.algorithm == 'fedavg':
    # ... FedAvg 评估逻辑

# 场景 B: FedRoD
elif args.algorithm == 'fedrod':
    # ... FedRoD 分离评估逻辑
    # - Head-P 评估 ID/IN-C
    # - Head-G 评估 OOD
```

**修改后:**
```python
# 删除了 --algorithm 参数

# 只保留 FedAvg 评估
print("评估模式: FedAvg (Global Model 承担所有任务)")
id_acc, tail_acc = evaluate_accuracy_metrics(...)
# ... 所有评估都基于 Global Model
```

**函数调用更新:**
```python
# 修改前
model, foogd_module = create_fedrod_model(...)

# 修改后
model, foogd_module = create_model(...)
```

### 2.5 server.py
**修改内容:**
- 将 `create_fedrod_model()` 调用改为 `create_model()`

---

## 三、代码架构变化

### 3.1 模型架构对比

**修改前 (FedRoD + Taxonomy):**
```
FedRoD_Model
├── Backbone (DenseNet)
│   └── features [batch, 1664]
├── Head_G (通用头)
│   ├── Linear(1664 → 512)
│   ├── ReLU
│   ├── Dropout(0.2)
│   └── Linear(512 → 54)
└── Head_P (个性化头)
    ├── Linear(1664 → 512)
    ├── ReLU
    ├── Dropout(0.2)
    └── Linear(512 → 54)

Loss = CrossEntropy + TaxonomyLoss + FOOGD_Loss
```

**修改后 (FedAvg + FOOGD):**
```
FedAvg_Model
├── Backbone (DenseNet)
│   └── features [batch, 1664]
└── Classifier (单一分类头)
    ├── Linear(1664 → 512)
    ├── ReLU
    ├── Dropout(0.2)
    └── Linear(512 → 54)

Loss = CrossEntropy + FOOGD_Loss
```

### 3.2 训练流程变化

**修改前:**
1. 客户端下载全局模型
2. 本地训练时:
   - Head-G: CE Loss + TaxonomyLoss + FOOGD Loss
   - Head-P: CE Loss
3. 上传 Head-G + Backbone 权重
4. 服务端聚合权重
5. 评估时:
   - ID/IN-C 准确率: 使用 Head-P
   - OOD 检测: 使用 Head-G 特征

**修改后:**
1. 客户端下载全局模型
2. 本地训练时:
   - Classifier: CE Loss + FOOGD Loss
3. 上传所有权重
4. 服务端聚合权重
5. 评估时:
   - 所有指标: 使用单一 Global Model

---

## 四、删除的代码统计

| 文件 | 删除行数 (约) | 主要内容 |
|------|--------------|---------|
| models.py | 70 | TaxonomyLoss 类 |
| models.py | 50 | FedRoD 双头改为单头 |
| client.py | 100 | taxonomy 参数和逻辑, evaluate_comprehensive |
| eval_utils.py | 20 | compute_hierarchical_error, 层级指标 |
| data_utils.py | 113 | build_taxonomy_matrix 函数 |
| train_federated.py | 70 | --algorithm 参数, FedRoD 评估分支 |
| **总计** | **~423 行** | |

---

## 五、保留的核心功能

✅ **FedAvg 联邦学习**
- 客户端-服务器架构
- 权重聚合 (FedAvg)
- Non-IID 数据支持 (Dirichlet 分布)

✅ **FOOGD OOD 检测**
- Score Model (评分网络)
- KSD Loss (核平滑散度)
- SM3D Loss (得分匹配)
- Langevin 动态采样

✅ **数据增强**
- 傅里叶增强
- 自动编码器

✅ **评估功能**
- ID 准确率
- Tail 类别准确率
- IN-C 泛化准确率
- Near-OOD / Far-OOD AUROC
- 混淆矩阵

---

## 六、测试建议

### 6.1 单元测试
```bash
cd experiments/plankton

# 测试模型创建
python -c "from models import create_model; model, foogd = create_model(); print('Model OK')"

# 测试客户端
python -c "from client import FLClient; print('Client OK')"

# 测试服务端
python -c "from server import FLServer; print('Server OK')"
```

### 6.2 集成测试 (小规模)
```bash
# 1 轮, 2 客户端, 快速验证
python train_federated.py \
    --num_clients 2 \
    --rounds 1 \
    --local_epochs 1 \
    --gpu_id 0
```

### 6.3 完整训练
```bash
# 完整配置
python train_federated.py \
    --num_clients 5 \
    --alpha 0.5 \
    --rounds 50 \
    --local_epochs 3 \
    --lr 0.001 \
    --gpu_id 0
```

---

## 七、注意事项

### 7.1 兼容性
- 所有检查点 (checkpoints) 需要重新训练,不兼容旧版本
- 旧的评估脚本需要更新模型输出处理

### 7.2 性能影响
- 内存占用减少 (删除了双头结构)
- 训练速度可能略有提升 (减少了 forward 计算)
- 模型精度可能下降 (移除了个性化头)

### 7.3 后续扩展
如需重新添加 FedRoD:
1. 参考 pFL-FOOGD-Plankton 原始项目
2. 恢复 FedRoD_Model 双头结构
3. 在 client.py 中添加 Head-P 保留逻辑
4. 在 train_federated.py 中添加 FedRoD 评估分支

---

## 八、修改文件列表

- [x] models.py - 模型架构简化
- [x] client.py - 客户端逻辑简化
- [x] server.py - 服务端函数调用更新
- [x] eval_utils.py - 评估函数简化
- [x] data_utils.py - 删除 taxonomy 工具函数
- [x] train_federated.py - 主训练脚本简化

---

## 九、验证清单

- [x] 所有 Taxonomy 导入已删除
- [x] 所有 FedRoD 双头逻辑已删除
- [x] 模型输出统一为 (logits, features)
- [x] 评估函数不包含 use_head_g 参数
- [x] 命令行参数不包含 --algorithm 和 --use_taxonomy
- [x] 代码中无 logits_p, head_p, Head-P 引用 (除注释外)

---

## 十、清理未使用的脚本

### 10.1 删除的文件

**utils/evaluate_head_p.py**
- **原因**: 专门评估 FedRoD 的 Head-P (个性化头) 性能
- **内容**: 加载 client_states，对比 Head-G 和 Head-P 准确率
- **删除日期**: 2025-12-27

**client.py 测试代码 (lines 297-335)**
- **原因**: 包含过时的测试代码，引用已删除的 `acc_g` 和 `acc_p` 指标
- **内容**:
  - 创建模型和数据加载器的测试代码
  - 调用 `evaluate()` 并打印 `acc_g` 和 `acc_p`
- **删除日期**: 2025-12-27

### 10.2 更新的文件

**utils/test_pipeline.py**
- **修改内容**: 更新为使用 FedAvg 单头架构
- **具体修改**:
  - `create_fedrod_model()` → `create_model()`
  - 移除 `head_g`、`head_p` 参数统计
  - 更新前向传播测试：`logits_g, logits_p, features` → `logits, features`
  - 更新打印信息：FedRoD模型 → FedAvg模型

**utils/evaluate_inc_robustness.py**
- **修改内容**: 更新模型创建函数调用
- **具体修改**: `create_fedrod_model()` → `create_model()`

**utils/verify_leakage.py**
- **修改内容**: 更新模型创建函数调用
- **具体修改**: `create_fedrod_model()` → `create_model()`

**eval_utils.py**
- **修改内容**: 简化 `compute_ood_scores()` 函数
- **具体修改**:
  - 移除 `use_head_g` 参数逻辑
  - 简化为直接取 `outputs[1]` 作为特征
  - 更新注释为 FedAvg 架构说明

### 10.3 保留的文件

以下脚本仍然有用，已保留且无需修改：

**数据集工具**:
- `split_dataset.py` - 数据集划分工具 (ID/Near-OOD/Far-OOD)

**可视化**:
- `visualize_experiments.py` - 绘制训练曲线和性能图表

**数据验证** (utils/):
- `check_md5_leakage.py` - MD5 哈希检查数据泄漏
- `check_overlap.py` - 文件名重叠检查
- `verify_leakage.py` - 综合数据泄漏验证 (已更新模型调用)

**测试** (utils/):
- `test_pipeline.py` - 单元测试框架 (已更新为 FedAvg)
- `test_foogd_pipeline.py` - FOOGD 模块集成测试

**评估** (utils/):
- `evaluate_inc_robustness.py` - IN-C 鲁棒性评估 (已更新模型调用)
- `README.md` - 工具文档 (需手动更新以反映 FedRoD 删除)

### 10.4 最终验证

运行以下命令验证清理完成：
```bash
# 应该返回 0 (无残留引用)
grep -r "FedRoD\|fedrod\|TaxonomyLoss\|taxonomy_matrix" experiments/plankton/*.py experiments/plankton/utils/*.py 2>/dev/null | grep -v "^Binary" | wc -l

# 应该返回 0 (无双头引用)
grep -r "head_p\|Head-P\|acc_g\|acc_p" experiments/plankton/*.py experiments/plankton/utils/*.py 2>/dev/null | grep -v "^Binary" | wc -l
```

---

## 十一、最终验证报告

### 11.1 代码清理验证

```bash
✅ FedRoD/Taxonomy 引用: 0 条
✅ 双头架构引用 (head_p/Head-P/acc_g/acc_p): 0 条
✅ 所有文件已更新为 FedAvg 单头架构
```

### 11.2 当前文件结构

**核心文件 (6个)**:
- [models.py](models.py) - FedAvg 单头模型 + FOOGD 模块
- [client.py](client.py) - 联邦学习客户端 (简化版)
- [server.py](server.py) - 联邦学习服务器
- [data_utils.py](data_utils.py) - 数据加载和 Dirichlet 划分
- [eval_utils.py](eval_utils.py) - 评估工具 (ID/OOD/IN-C)
- [train_federated.py](train_federated.py) - 主训练脚本

**工具脚本 (2个)**:
- [split_dataset.py](split_dataset.py) - 数据集划分工具
- [visualize_experiments.py](visualize_experiments.py) - 可视化脚本

**Utils 目录 (6个)**:
- check_md5_leakage.py - 数据泄漏检查
- check_overlap.py - 文件重叠检查
- evaluate_inc_robustness.py - IN-C 鲁棒性评估
- test_foogd_pipeline.py - FOOGD 集成测试
- test_pipeline.py - 单元测试框架
- verify_leakage.py - 综合泄漏验证

**文档 (2个)**:
- [删除taxonomy和fedrod.md](删除taxonomy和fedrod.md) - 本文档
- utils/README.md - 工具文档

### 11.3 功能验证清单

- [x] 模型架构: FedRoD 双头 → FedAvg 单头
- [x] 模型输出: (logits_g, logits_p, features) → (logits, features)
- [x] 损失函数: CE + Taxonomy + FOOGD → CE + FOOGD
- [x] 评估指标: 移除 hier_acc, hier_error, acc_g, acc_p
- [x] 命令行参数: 移除 --algorithm, --use_taxonomy
- [x] 函数调用: create_fedrod_model() → create_model()
- [x] 所有测试脚本已更新为 FedAvg 架构
- [x] 文档已更新

### 11.4 下一步工作

现在代码已清理为纯净的 FedAvg + FOOGD 架构，可以开始集成 Fed-ViM:

1. **在 server.py 中添加 GSA 聚合**:
   - 统计量收集: Σz, Σzzᵀ
   - 全局协方差计算
   - PCA 子空间提取

2. **在 client.py 中添加 GSA Loss**:
   - 接收全局子空间 P, μ
   - 计算子空间投影损失
   - 返回本地统计量

3. **更新评估脚本**:
   - ViM Score 计算
   - OOD 检测性能评估

---

**清理完成时间**: 2025-12-27
**代码状态**: ✅ 已验证无残留引用
**准备就绪**: 可以开始 Fed-ViM 集成工作

**修改者**: Claude Code
**项目**: Fed-ViM/experiments/plankton
