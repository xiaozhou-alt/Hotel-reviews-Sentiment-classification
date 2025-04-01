import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


# 配置参数
bert_model_name = 'Qunar/bert_base'
image_size = 224
batch_size = 16
epochs = 5
learning_rate = 2e-5
cache_dir = 'your_cache_directory'  # 替换为实际缓存目录
model_save_path = 'model.pth'


# 数据加载
def load_data():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构造完整文件路径
    multi_path = os.path.join(script_dir, 'multi.csv')
    object_path = os.path.join(script_dir, 'object.csv')
    picture_path = os.path.join(script_dir, 'picture.csv')
    review_path = os.path.join(script_dir, 'review.csv')
    
    # 加载数据
    multi_df = pd.read_csv(multi_path)
    object_df = pd.read_csv(object_path)
    picture_df = pd.read_csv(picture_path)
    review_df = pd.read_csv(review_path)
    
    return multi_df, object_df, picture_df, review_df


# 数据预处理
def preprocess_data(multi_df, review_df):
    # 合并数据
    if 'unique_id' in multi_df.columns and 'unique_id' in review_df.columns:
        # 字段映射逻辑
        # 处理multi_df
        multi_df['sentiment'] = multi_df['small_score'].apply(
            lambda x: 1 if 's5' in x else 2 if 's4' in x else 3)
        multi_df['image_id'] = multi_df['unique_id'].astype(str) + '_multi'
        
        # 处理review_df
        if 'small_score' in review_df.columns:
            review_df['sentiment'] = review_df['small_score'].apply(
                lambda x: 1 if 's5' in x else 2 if 's4' in x else 3)
        review_df['image_id'] = review_df['unique_id'].astype(str) + '_review'
        
        # 选择保留字段
        keep_columns = ['unique_id', 'txt', 'sentiment', 'image_id']

        # 垂直合并数据集
        merged_df = pd.concat([
            multi_df[keep_columns],
            review_df[keep_columns]
        ], ignore_index=True)
        print("合并后的列名:", merged_df.columns.tolist())
        
        # 自动识别有效文本列
        text_columns = [col for col in merged_df.columns if any(keyword in col.lower() for keyword in ['txt', 'text', 'content', 'review'])]
        # 统一重命名文本列为text_content
        if 'txt' in merged_df.columns:
            merged_df.rename(columns={'txt': 'text_content'}, inplace=True)
        
        print("合并后的数据集维度:", merged_df.shape)
    else:
        print("数据缺少unique_id列，无法合并")
        return None

    # 缺失值处理
    merged_df.dropna(inplace=True)

    # 验证必需字段存在
    required_columns = ['text_content', 'image_id', 'sentiment']
    missing_columns = [col for col in required_columns if col not in merged_df.columns]
    if missing_columns:
        raise ValueError(f"合并数据缺少必要字段: {missing_columns}")

    # 划分训练集和测试集
    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
    return train_df, test_df


# 自定义数据集类
class QunarDataset(Dataset):
    def __init__(self, df, tokenizer, transform):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if 'text_content' not in row:
            raise KeyError("合并后的数据缺少有效的文本列，请检查数据合并逻辑")
        text = row['text_content']
        image_path = os.path.join('../pic', row['image_id'] + '.jpg')
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except FileNotFoundError:
            image = torch.zeros(3, image_size, image_size)

        label = row['sentiment']

        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'text_content': text,
            'image_path': image_path
        }


# 多模态模型
class MultiModalModel(torch.nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            bert_model_name,
            num_labels=4,
            cache_dir=cache_dir,
            mirror="https://mirror.sjtu.edu.cn/huggingface-models/"
        )
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_model.fc = torch.nn.Linear(self.image_model.fc.in_features, 4)
        self.fusion_layer = torch.nn.Linear(8, 4)

    def forward(self, input_ids, attention_mask, image):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask).logits
        image_output = self.image_model(image)
        combined = torch.cat((text_output, image_output), dim=1)
        output = self.fusion_layer(combined)
        return output


# 训练模型
# 修改后的训练函数
def train_model(model, train_dataloader, optimizer, criterion, device, epochs):
    model.train()
    
    # 创建总进度条
    progress_bar = tqdm(range(epochs), desc="总训练进度", position=0)
    
    for epoch in range(epochs):
        # 创建epoch进度条
        epoch_progress = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            position=1
        )
        
        total_loss = 0
        for batch in epoch_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 实时更新进度条描述
            epoch_progress.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(epoch_progress.n+1):.4f}"
            })

        # 更新总进度条
        avg_loss = total_loss / len(train_dataloader)
        progress_bar.set_postfix({'epoch_loss': f"{avg_loss:.4f}"})
        progress_bar.update(1)
        
        # 关闭epoch进度条
        epoch_progress.close()
    
    progress_bar.close()


# 评估模型
def evaluate_model(model, test_dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, image)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    print(classification_report(true_labels, predictions))
    return predictions, true_labels


# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


# 预测推理
def predict(model, tokenizer, transform, text, image_path, device):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        image = torch.zeros(1, 3, image_size, image_size).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image)
        _, prediction = torch.max(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]

    sentiment_mapping = {0: '未涉及', 1: '积极', 2: '中性', 3: '消极'}
    sentiment_label = sentiment_mapping[prediction.item()]
    
    # 创建带名称的概率分布
    prob_details = {
        sentiment_mapping[0]: f"{probabilities[0]:.2%}",
        sentiment_mapping[1]: f"{probabilities[1]:.2%}",
        sentiment_mapping[2]: f"{probabilities[2]:.2%}",
        sentiment_mapping[3]: f"{probabilities[3]:.2%}"
    }
    
    return sentiment_label, prob_details, text, image_path


# 可视化评估结果
def visualize_results(predictions, true_labels):
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    confusion_matrix = pd.crosstab(pd.Series(true_labels, name='真实情感'),
                                   pd.Series(predictions, name='预测情感'))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('混淆矩阵')
    plt.show()


# 主函数
def main():
    multi_df, object_df, picture_df, review_df = load_data()
    train_df, test_df = preprocess_data(multi_df, review_df)

    tokenizer = AutoTokenizer.from_pretrained(
        bert_model_name,
        cache_dir=cache_dir,
        mirror="https://mirror.sjtu.edu.cn/huggingface-models/"
    )
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = QunarDataset(train_df, tokenizer, transform)
    test_dataset = QunarDataset(test_df, tokenizer, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    train_model(model, train_dataloader, optimizer, criterion, device, epochs)

    # 评估模型
    predictions, true_labels = evaluate_model(model, test_dataloader, device)

    # 可视化评估结果
    visualize_results(predictions, true_labels)

    # 保存模型
    save_model(model, model_save_path)

    # 预测示例
    # 使用数据集中的实际样本来测试
    sample = test_df.iloc[0]
    test_text = sample['text_content']
    test_image_path = os.path.join('../pic', sample['image_id'] + '.jpg')
    sentiment_label, prob_details, text_content, img_path = predict(model, tokenizer, transform, test_text, test_image_path, device)
    
    print(f"\n识别结果:")
    print(f"文本内容: {text_content}")
    print(f"图片路径: {img_path}")
    print(f"预测情感: {sentiment_label}")
    print("概率分布:")
    for category, prob in prob_details.items():
        print(f"  {category}: {prob}")


if __name__ == "__main__":
    main()
    