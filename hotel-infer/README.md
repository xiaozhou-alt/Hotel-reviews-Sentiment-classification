# 项目介绍
本项目基于Qunar酒店评论多模态数据集，目标是**通过文本信息预测评论涉及的酒店服务方面（Aspect Category）是否相关（0/1二分类）**。  
数据来自去哪儿网爬取，包含3,518条评论及多种形式的标注信息（文本、图片、目标检测等）

# 问题一介绍
- **任务一（文本部分）**：用文本（review）预测酒店评论的主题。
- **任务二（图像部分）**：用图片预测酒店评论的主题。

只判断和某主题是否有关，不判断情感方向。
| 中文类别 | 说明 |
|----------|------|
| 区位     | 酒店地理位置、交通便利性等 |
| 餐饮     | 早餐、餐厅等评价          |
| 房间设施 | 客房内部设施（床、空调、电视等）|
| 娱乐设施 | 游泳池、健身房、儿童区等     |
| 店内设施 | 电梯、大堂、走廊等公共设施    |
| 服务     | 前台服务、服务态度、响应速度等 |


# BERT

### 什么是BERT？

BERT，全称 **Bidirectional Encoder Representations from Transformers**，是一种由 Google 在 2018 年提出的**基于Transformer结构的预训练语言模型**。

它的核心优势有两个：

1. **双向理解文本（Bidirectional）**  
    相比于以往的RNN/LSTM从左到右的顺序建模，BERT使用Transformer的自注意力机制，同时考虑上下文左右两侧的词语，理解更全面。
    
2. **预训练+微调（Pretraining + Fine-tuning）机制**  
    BERT首先在超大规模语料库上进行通用语言学习（预训练），然后可以针对各种下游任务（分类、问答、抽取）进行少量数据微调，效果非常好。

![[Pasted image 20250406224627.png]]
### bert_base_chinese

`bert-base-chinese` 是一个对**简体中文文本**进行预训练的BERT模型，相比英文BERT，`bert-base-chinese` 能处理中文短语、歧义、多义词等问题。

![[Pasted image 20250406224023.png]]
### BERT的核心结构Transformer 编码器（Encoder）

BERT 基于的是 **Transformer 的 Encoder部分**，它是目前最主流的文本理解架构。每层包括：

- **多头自注意力机制（Multi-head Self-Attention）**
- **前馈神经网络（Feed Forward Layer）**
- 残差连接 + LayerNorm


             多头注意力机制结构示意图
┌────────────────────────────────────────────┐
│                   输入序列 ：                                                                         │
│       "酒店" "服务" "特别" "好" "早餐"                                                     │
└────────────────────────────────────────────┘
             ↓             ↓            ↓
        Head 1         Head 2       Head 3
       (关注主宾结构)   (关注情感表达)  (关注实体位置)
             ↓             ↓            ↓
        Linear层拼接 + 投影 → 输出综合表示

本实验是堆叠了12层**Transformer Encoder结构**，每层都使用多头注意力，可以深度捕捉“服务特别好”“早餐丰富”“交通便利”等关键词之间的**依赖关系**，帮助模型更准确判断评论中提及了哪些方面（Aspect）
####  什么是多头注意力？

> 多头注意力机制允许模型从多个子空间中“关注”词与词之间的关系。

比如句子“酒店服务很好”中，模型可以：

- Head 1：理解“服务”与“酒店”之间的主谓关系；
- Head 2：理解“服务”与“好”之间的情感关系；
- Head 3：理解“酒店”在整个句子中的位置和角色。

每个注意力头计算方式如下：

`Attention(Q, K, V) = softmax(QKᵀ / √d) × V`

其中 Q（Query）、K（Key）、V（Value）都来自输入词向量。
### 模型训练
BERT本身不是直接用于分类的，它是在两个预训练任务上先学好语言能力的：
#### 1. Masked Language Model (MLM)

- 随机把句子中一些词遮住（用 [MASK] 替代），让模型去“猜”被遮掉的词是什么。
  
    > 比如：“酒店位置[MASK]好”，模型要预测“很”。
#### 2. Next Sentence Prediction (NSP)

- 给模型两个句子，判断后一句是不是前一句的真实上下文。
# 问题一第一部分-文本-对review用bert模型进行预测

###  模型在本项目中的应用方式

我们将每条酒店评论作为输入文本，使用BERT提取其[CLS]语义向量，接一个线性层预测其是否涉及各Aspect类别，多分类二标签。

把每条酒店评论输入`bert-base-chinese`模型 → 提取 `[CLS]` 向量 → 接一个输出为6维的Sigmoid分类器 → 得到6个概率 → 判断每个方面是否提及（0/1）（最终每个类别根据概率是否大于0.5判断是否“提及”）


## 问题一代码具体解释流程

### 1. 数据预处理

**输入：**
原始文本：`review_words`列（列表形式，需合并成字符串）

**标签：**

- `区位`, `餐饮`, `房间设施`, `娱乐设施`, `店内设施`, `服务` 六列
- 原始标签为 {0,1,2,3}，我们统一为二分类（0：未涉及，1：涉及）

def label_binarize(x):
    return 0 if x == 0 else 1
df['review'] = df['review_words'].apply(lambda x: ''.join(eval(x)))


### 2. 构建Dataset与Dataloader

利用`transformers`的`BertTokenizer`将文本转为模型输入格式，创建`Dataset`和`DataLoader`：

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
encoding = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=128,
    truncation=True,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)

---

### 3. 模型定义与训练

使用`BertModel`搭建一个自定义分类器：

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return self.out(self.drop(pooled))
### 损失函数和优化器

- 多标签二分类问题 → `BCEWithLogitsLoss`
  
- 优化器：`AdamW`（适合Transformer）
  

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

### 4. 模型训练与评估

- 将每条评论输入模型，预测6个标签是否相关
  
- 使用`F1-score`和`Accuracy`作为评估指标
  

---

### 5. 推理（预测新评论）

def predict(text):
    encoding = tokenizer.encode_plus(...)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        pred = torch.sigmoid(output)
    return pred > 0.5

#### **代码中的 ResNet-101 架构**

1. **基础结构**
   基于 PyTorch 官方的 `torchvision.models.resnet101` 构建，加载了 ImageNet 预训练权重 (`ResNet101_Weights.IMAGENET1K_V1`)，包含以下核心组件：

   - **初始卷积层**：
     `conv1 → bn1 → relu → maxpool`（输出 64 通道）

   - 

     残差块组

     ：

     - `layer1`（3 个残差块，输出 256 通道）
     - `layer2`（4 个残差块，输出 512 通道）
     - `layer3`（23 个残差块，输出 1024 通道）
     - `layer4`（3 个残差块，输出 2048 通道）

2. **关键修改**
   在基础 ResNet-101 上添加了 ​**​通道注意力模块​**​（`ChannelAttention`）：

   - 

     插入位置

     ：

     - `layer1` 后（256 通道）
     - `layer2` 后（512 通道）
     - `layer4` 后（2048 通道）

   - 

     注意力计算逻辑

     ：

     对每个通道执行自适应平均池化（AdaptiveAvgPool）和自适应最大池化（AdaptiveMaxPool），通过全连接层生成通道权重，公式为：

     python

     复制

     ```python
     x = x * (Sigmoid(FC(AvgPool(x)) + FC(MaxPool(x))))
     ```

3. **多任务输出头**

   - **主题分类头**（8 个二分类任务）：
     `GlobalAvgPool → FC(2048→512) → ReLU → Dropout → FC(512→8) → Sigmoid`
   - **情感分类头**（三分类任务）：
     `GlobalAvgPool → FC(2048→512) → ReLU → Dropout → FC(512→3)`

4. **参数冻结策略**

   - **冻结层**：`conv1`、`bn1`、`maxpool`、`layer1`、`layer2`
   - **微调层**：`layer3`、`layer4` 及所有新增模块（注意力机制、分类头）

------

#### **应用场景**

1. **图像特征提取**

   - **输入**：224×224 RGB 图像

   - 

     预处理

     ：

     python

     复制

     ```python
     transforms.Resize(224) → ToTensor() → Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
     ```

   - 

     输出

     ：

     - **主题分类**：8 个酒店场景主题（如餐饮、房间设施等）的概率
     - **情感分类**：积极/中性/消极三分类概率

2. **实际应用**

   - **多模态分析**：结合文本模型（BERT）进行酒店评论的图文联合分析

   - 

     功能支持

     ：

     - 识别图像中的酒店设施主题
     - 判断用户上传图片的情感倾向（如客房照片的情感分析）

```text
Input Image
    │
    ▼
ResNet-101 基础特征提取层
    │（含通道注意力模块）
    ▼
Global Average Pooling
    │
    ├───▶ 主题分类头 → 8 个二分类结果
    │
    └───▶ 情感分类头 → 三分类结果
```

