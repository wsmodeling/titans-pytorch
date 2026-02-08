# Documentation Index

Quick reference guide to all documentation in this repository.

## 📚 Main Documentation

### [CONVERSATION_SUMMARY.md](CONVERSATION_SUMMARY.md) ⭐ **START HERE**
**完整对话记录** - 包含所有技术细节、原理解释和实现过程

涵盖内容：
- CUDA 错误修复过程
- MAC Transformer 详细架构解析
- Neural Memory 完整工作原理
- KDA Memory 实现细节
- Memory 切换功能实现
- 关键概念解释（Teacher Forcing, Associative Scan 等）

**适合**:
- 想深入理解整个系统
- 需要技术背景知识
- Debug 或扩展功能时参考

---

### [MEMORY_SWITCHING_GUIDE.md](MEMORY_SWITCHING_GUIDE.md) ⚡ **QUICK START**
**快速使用指南** - 如何切换 Neural Memory 和 KDA Memory

涵盖内容：
- 一行代码切换 memory 类型
- 配置参数说明
- 性能特点对比
- 常见问题解决

**适合**:
- 快速开始使用
- 只想知道怎么切换
- 不需要深入技术细节

---

### [KDAMEMORY_README.md](KDAMEMORY_README.md) 📖 **KDA SPECIFIC**
**KDA Memory 专项文档**

涵盖内容：
- KDA Memory 基础概念
- API 使用方法
- 与 Neural Memory 的对比
- 配置参数详解
- 性能优化建议

**适合**:
- 专门使用 KDA Memory
- 理解 KDA 算法原理
- 性能调优

---

## 🧪 Test & Demo Files

### [test_kda_memory.py](test_kda_memory.py)
KDA Memory 单元测试
- 基础前向传播
- 增量推理
- 与 NeuralMemory 兼容性
- 梯度反向传播
- Recurrent vs Chunk 一致性

运行: `python test_kda_memory.py`

### [demo_memory_switching.py](demo_memory_switching.py)
Memory 类型切换演示
- 创建两种 memory 类型的模型
- 参数数量对比
- 前向传播验证

运行: `python demo_memory_switching.py`

---

## 📝 Code Files

### Core Implementation

#### [titans_pytorch/kda_memory.py](titans_pytorch/kda_memory.py)
KDA Memory 核心实现
- `naive_recurrent_kda()` - 递归版本
- `naive_chunk_kda()` - 分块并行版本
- `KDAMemory` - 主模块类
- `create_kda_memory_for_mac()` - 工厂函数

#### [titans_pytorch/neural_memory.py](titans_pytorch/neural_memory.py)
Neural Memory 核心实现
- Test-time training 机制
- 权重更新和检索

#### [titans_pytorch/mac_transformer.py](titans_pytorch/mac_transformer.py)
MAC Transformer 实现
- 支持两种 memory 类型
- Segmented attention
- Longterm & Persistent memory

#### [train_mac.py](train_mac.py)
训练脚本
- **第 46 行**: `MEMORY_TYPE` flag 🔥
- **第 70 行**: KDA 配置参数
- **第 108-128 行**: Memory 模型创建

---

## 🔍 Quick Reference

### 切换 Memory 类型

```python
# In train_mac.py, line ~46
MEMORY_TYPE = 'neural'  # TTT-based Neural Memory
# OR
MEMORY_TYPE = 'kda'     # KDA Linear Attention
```

### 关键配置参数

**Neural Memory**:
- `NEURAL_MEMORY_DEPTH = 2` - MLP 深度
- `NEURAL_MEM_BATCH_SIZE = 128` - 权重更新频率
- `NEURAL_MEM_SEGMENT_LEN = 4` - Chunk 大小

**KDA Memory**:
- `KDA_CHUNK_SIZE = 32` - 分块大小
- `KDA_USE_CHUNK = True` - 使用分块模式

**Common**:
- `WINDOW_SIZE = 32` - Attention 窗口
- `NUM_LONGTERM_MEM = 4` - 长期记忆 tokens
- `NUM_PERSIST_MEM = 4` - 持久记忆 tokens

### 参数数量对比

| Memory Type | Parameters | Difference |
|------------|-----------|------------|
| Neural Memory | 19,980,207 | Baseline |
| KDA Memory | 20,056,377 | +76,170 (+0.38%) |

---

## 📊 Architecture Diagrams

### Memory Flow (Neural Memory)
```
Input → Store → Compute Surprises → Accumulate (Scan) → Updates
                                                            ↓
Output ← Retrieve ← Query State ←──────────────────── State Matrix
```

### Memory Flow (KDA Memory)
```
Input → Q/K/V/Gates/Beta → Chunk KDA → Recurrent Updates → Output
                              ↓
                         State Matrix [B,H,D,D]
```

### MAC Transformer Layer
```
Input
  ↓
[Optional] Neural Memory / KDA Memory
  ↓
Segmented Attention + Persistent Memory
  ↓
FeedForward
  ↓
Output
```

---

## 🐛 Common Issues & Solutions

### Issue: CUDA illegal memory access
**File**: train_mac.py
**Line**: 73
**Solution**: `USE_ACCELERATED_SCAN = False`

### Issue: Dimension mismatch in memory model
**Cause**: Template created with wrong dim
**Solution**: Use `dim_head` (64) for template, MAC will recreate with transformer dim (384)

### Issue: Beta shape error
**Cause**: Expected `[B,T,H,D]` but should be `[B,T,H]`
**Solution**: Beta is per-head scalar, not per-dimension vector

---

## 📖 Glossary

### Key Terms

- **TTT (Test-Time Training)**: 在前向传播时更新权重来存储记忆
- **Teacher Forcing**: 训练时使用真实标签而非模型预测
- **Associative Scan**: 并行化线性 RNN 的算法
- **Longterm Memory Tokens**: 插入序列中的可学习桥梁 tokens
- **Persistent Memory**: 每层独立的全局可见 KV 对
- **KDA**: Kimi Dynamic Attention，线性注意力机制

---

## 🔗 External Resources

### Papers
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2408.06654)
- [Kimi Linear Attention](https://arxiv.org/abs/2410.23029)
- [Test-Time Training](https://arxiv.org/abs/1909.13231)

### Code
- [Titans PyTorch](https://github.com/lucidrains/titans-pytorch)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

---

## 💡 Navigation Tips

1. **First time user?** → Start with [MEMORY_SWITCHING_GUIDE.md](MEMORY_SWITCHING_GUIDE.md)
2. **Want deep understanding?** → Read [CONVERSATION_SUMMARY.md](CONVERSATION_SUMMARY.md)
3. **Using KDA specifically?** → See [KDAMEMORY_README.md](KDAMEMORY_README.md)
4. **Need to debug?** → Check "Common Issues" section above
5. **Extending the code?** → Refer to [CONVERSATION_SUMMARY.md](CONVERSATION_SUMMARY.md) Section 3 & 4

---

**Last Updated**: 2025-02-08
**Project**: Titans-PyTorch with KDA Memory Integration
**Maintainer**: Keep this index updated as documentation evolves
