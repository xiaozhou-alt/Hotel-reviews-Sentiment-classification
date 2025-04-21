import pandas as pd
import torch

df = pd.read_csv('data//review.csv')
df.head()

def label_binarize(x):
    return 0 if x==0 else 1

aspects = ['区位', '餐饮', '房间设施', '娱乐设施', '店内设施', '服务']
for col in aspects:
    df[col] = df[col].apply(label_binarize)

df['review'] = df['review_words'].apply(lambda x: ''.join(eval(x)))

train_df = df[['review'] + aspects]

#数据集划分
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

# 检查GPU可用性并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自动选择GPU如果可用
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df['review'].values
        self.labels = df[aspects].values.astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        # 初始化BertClassifier类，并传入类别数n_classes
        super(BertClassifier, self).__init__()
        # 调用父类初始化方法
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 从预训练模型中加载Bert模型
        self.drop = nn.Dropout(p=0.3)
        # 定义Dropout层，用于防止过拟合，p为0.3
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        output = self.drop(pooled_output)
        return self.out(output)

# 直接使用之前定义的BertClassifier
model = BertClassifier(len(aspects)).to(device)  # 统一使用device变量

# 初始化优化器和损失函数
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)



# 数据集初始化
train_dataset = ReviewDataset(train_data, tokenizer)
val_dataset = ReviewDataset(val_data, tokenizer)

# 使用pin_memory加速数据转移到GPU
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, pin_memory=True)



def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    losses = []

    for batch in data_loader:
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)

def eval_model(model, data_loader):
    model.eval()
    predictions = []
    real_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs).cpu().numpy()

            predictions.extend(preds)
            real_labels.extend(labels)

    predictions = np.array(predictions) >= 0.5
    #accuracy = accuracy_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions, normalize=True, sample_weight=None, multilabel=True)
    f1 = f1_score(real_labels, predictions, average='macro')

    return accuracy, f1

# 开始训练
epochs = 3
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_acc, val_f1 = eval_model(model, val_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')

def predict(text, model, tokenizer):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.sigmoid(outputs).cpu().numpy()[0]

    return {aspect: int(pred >= 0.5) for aspect, pred in zip(aspects, preds)}

# 示例：
text = "酒店位置很好，离景点很近，但餐饮一般。"
result = predict(text, model, tokenizer)
print(result)
