# CMU 11-868: LLM System Homework

## Assignment 1: Minitorch Framework  
- 实现了自动微分系统，包括拓扑排序和反向传播算法
- 构建了神经网络架构，完成了线性层、多层感知机和情感分类网络
- 实现了训练循环和交叉熵损失函数，在SST-2数据集上达到75%以上准确率

## Assignment 2: CUDA Programming  
- 实现了CUDA内核的map、zip、reduce和矩阵乘法操作
- 通过共享内存优化和分块技术提升GPU计算效率
- 集成CUDA后端，加速张量运算

## Assignment 3: Transformer Architecture  
- 实现了GPT-2解码器架构，包括多头注意力、前馈网络和层归一化
- 完成了机器翻译pipeline，在IWSLT数据集上训练
- 生成文本并计算BLEU分数, 10个epochs后达到20

## Assignment 4: Transformer CUDA Acceleration  
- 优化Softmax和LayerNorm的CUDA内核
- 实现融合核函数，减少内存访问开销
- 在Transformer中集成优化核，获得约1.1倍整体加速

## Assignment 5: Distributed Training and Parallelism  
- 实现数据并行训练，支持梯度聚合
- 实现流水线并行，优化GPU资源利用
- 比较不同并行策略的训练时间和吞吐量，绘制图表

## Assignment 6: Advanced Training and Inference Systems  
- 使用DeepSpeed ZeRO和LoRA进行大模型高效训练
- 基于SGLang实现高性能推理，利用RadixAttention和压缩有限状态机优化












