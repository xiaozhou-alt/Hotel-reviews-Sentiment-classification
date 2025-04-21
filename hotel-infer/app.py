from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.resnet import ResNet101_Weights  # 添加权重枚举
import torch
from torch import nn
from flask import send_from_directory
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from werkzeug.utils import secure_filename  # 安全文件名处理
from flask_cors import CORS  # 跨域支持
import logging  # 日志记录
from transformers import set_seed


set_seed(42)
os.environ['TRANSFORMERS_OFFLINE'] = '1' 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).flatten(1))  # 更安全的展平方式
        max_out = self.fc(self.max_pool(x).flatten(1))
        return x * (avg_out + max_out).unsqueeze(-1).unsqueeze(-1)  # 更直观的维度扩展


class ResNet101Att(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        
        # 完整的特征提取部分
        self.features = nn.Sequential(
            base_model.conv1,           # [batch, 64, 112, 112]
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,         # [batch, 64, 56, 56]
            base_model.layer1,          # [batch, 256, 56, 56]
            ChannelAttention(256),      # 插入通道注意力
            base_model.layer2,          # [batch, 512, 28, 28]
            ChannelAttention(512),      # 插入通道注意力
            base_model.layer3,          # [batch, 1024, 14, 14]
            base_model.layer4,          # [batch, 2048, 7, 7]
            ChannelAttention(2048)      # 插入通道注意力
        )
        
        # 多任务输出头
        self.avgpool = base_model.avgpool  # [batch, 2048, 1, 1]
        
        # 主题分类头（8个二分类）
        self.theme_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8),
            nn.Sigmoid()
        )
        
        # 情感分类头（三分类）
        self.sentiment_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

        # 冻结参数策略
        for param in base_model.parameters():
            param.requires_grad = False
        for param in base_model.layer3.parameters():
            param.requires_grad = True
        for param in base_model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return {
            'themes': self.theme_head(x),
            'sentiment': self.sentiment_head(x)
        }


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = None  # 延迟初始化
        self.drop = nn.Dropout(p=0.3)
        self.out = None
        self.n_classes = n_classes

    def init_bert(self, model_path):
        self.bert = BertModel.from_pretrained(model_path)
        self.out = nn.Linear(self.bert.config.hidden_size, self.n_classes)
        return self
    
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # 获取[CLS]标记的隐藏状态
        
        return self.out(self.drop(pooled_output))

# 修改后的加载函数
def load_bert_model(checkpoint_path, device='cpu'):
    model_path = "model/bert_base_chinese"
    
    # 初始化空模型
    aspects = ['区位', '餐饮', '房间设施', '娱乐设施', '店内设施', '服务']
    model = BertClassifier(len(aspects))
    
    try:
        # 分步加载
        model.init_bert(model_path)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device),
            strict=False
        )
        tokenizer = BertTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")
    
    return model.to(device), tokenizer

# 初始化模型
def load_model(checkpoint_path, device='cpu'):
    # 初始化模型结构（必须与训练时完全相同）
    model = ResNet101Att()  # 确保ResNet101Att类已正确定义
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 检查checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:  # 如果是直接保存的state_dict
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"模型文件 {checkpoint_path} 不存在")
    
    model = model.to(device)
    model.eval()  # 现在可以正确调用eval()
    return model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_model = load_model("model/best_sentiment_acc_51.95.pth", device)
text_model, tokenizer = load_bert_model("model/bert_model.pth", device)


# 文件类型检查
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 预测处理
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = image_model(tensor)
    
    # 处理输出结果
    theme_probs = torch.sigmoid(outputs['themes'])[0].tolist()
    sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0].tolist()
    
    theme_names = ['区位','餐饮','房间设施','娱乐设施','店内设施','服务','风格化','其他']
    main_theme = theme_names[theme_probs.index(max(theme_probs))]
    
    sentiment_labels = ['消极 (-1)', '中性 (0)', '积极 (1)']
    sentiment = sentiment_labels[sentiment_probs.index(max(sentiment_probs))]
    
    return {
        'theme_probs': dict(zip(theme_names, [f"{p*100:.1f}%" for p in theme_probs])),
        'main_theme': main_theme,
        'sentiment': sentiment,
        'sentiment_probs': {
            '消极': f"{sentiment_probs[0]*100:.1f}%",
            '中性': f"{sentiment_probs[1]*100:.1f}%",
            '积极': f"{sentiment_probs[2]*100:.1f}%"
        }
    }

def predict_text(text):
    # 文本编码
    inputs = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # 显式获取张量
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        # 正确调用方式
        outputs = text_model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    aspects = ['区位', '餐饮', '房间设施', '娱乐设施', '店内设施', '服务']
    return {k: float(v) for k, v in zip(aspects, probs)}

def multi_predict(image_path, text):
    # 并行预测
    image_result = predict_image(image_path)
    text_result = predict_text(text)
    
    # 结果融合
    combined = {
        'image': image_result,
        'text': text_result,
        'combined_theme': _combine_themes(
            image_result['main_theme'],
            text_result
        )
    }
    return combined

def _combine_themes(image_theme, text_probs):
    # 示例融合逻辑：取文本预测最高分项与图像主题比较
    text_main = max(text_probs, key=text_probs.get)
    return text_main if text_probs[text_main] > 0.7 else image_theme


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回空响应

@app.route('/predict', methods=['POST'])
def handle_predict():
    try:
        # 调试日志：记录请求基本信息
        app.logger.info(f"收到请求，表单数据: {request.form}，文件: {request.files}")
        
        # 获取文件对象
        file = request.files.get('file')
        text = request.form.get('text', '')

        # 双重校验
        if not file and not text:
            app.logger.warning("未收到任何有效输入")
            return jsonify({'error': '必须提供文本或图片'}), 400

        # 处理图片上传
        img_path = None
        if file and file.filename != '':
            if not allowed_file(file.filename):
                app.logger.error(f"非法文件类型: {file.filename}")
                return jsonify({'error': '文件类型不合法'}), 400
                
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            file.save(img_path)
            app.logger.info(f"文件保存成功: {img_path}")

        # 执行预测
        result = {}
        if text:
            result['text'] = predict_text(text)
            app.logger.debug(f"文本预测完成: {text}")
        if img_path:
            result['image'] = predict_image(img_path)
            app.logger.debug(f"图片预测完成: {img_path}")

        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)