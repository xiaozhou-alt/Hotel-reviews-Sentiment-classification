## **酒店多模态评论主题及情感预测**

### 文件简介：

```bash
hotel-infer/
├── model/
│   ├── bert_code_lines/
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   │   └──tokenizerconfig.json
│   │   └──tokenizer.json
│   │   └──vocab.txt
│   └── best_sentimentacc_51.95.pth
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js            
│   └── uploads/
├── templates/
│   └── index.html
├── app.py                   # Flask主程序
├── README.md                # 项目说明文档
├── README-src.md            # 代码专项说明
└── requirements.txt          # 依赖库列表
```

1. **app.py**
   - Flask Web应用入口
   - 包含路由处理、文件上传、模型预测逻辑
   - 默认监听端口：`5000`
2. **model/best_sentiment_acc_51.26.pth**
   - 训练完成的PyTorch模型文件
   - 需确保与`ResNet101Att`模型类结构完全匹配
3. **templates/index.html**
   - 基于Bootstrap的响应式网页模板
   - 包含文件上传表单和结果展示区域
4. **static/***
   - 前端三件套（HTML/CSS/JS）的存储位置
   - `uploads/`目录会在首次运行时自动创建

### 运行

```bash
python app.py
```

