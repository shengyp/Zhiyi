## 代码结构
```
Suicide_Risk  
 ┣ evidence  
 ┃ ┗ slang_emoji_dict.json # 隐语证据库  
 ┣ output # 输出数据位置
 ┃ ┣ 20260424_202034 # 包含一组数据处理流程的完整数据
 ┃ ┃ ┣ stage1_filtered.json # 数据筛选与隐语识别示例输出
 ┃ ┃ ┣ stage2_translation_log.json # 隐语翻译示例输出
 ┃ ┃ ┗ final_semantic_completion.json # 帖文重写示例输出
 ┃ ┣ validation # 不同条件下四个模型的验证数据
 ┃ ┃ ┣ validation_report_20260426_232233 # 无数据增强，全字段输入
 ┃ ┃ ┣ validation_report_20260427_003932 # 分折数据增强，全字段输入
 ┃ ┃ ┣ validation_report_20260427_005757 # 无数据增强，仅输入content
 ┃ ┃ ┗ validation_report_20260427_012126 # 无数据增强，输入content与literal meaning
 ┣ spider  
 ┃ ┣ weiboSpider # 微博爬虫  
 ┃ ┃ ┣ ...  
 ┃ ┃ ┣ weibo  
 ┃ ┃ ┃ ┗ 我们金属板是这样的 # 抓取的用户数据，文件夹名为用户名  
 ┃ ┃ ┃ ┃ ┣ img  
 ┃ ┃ ┃ ┃ ┃ ┣ 原创微博图片  
 ┃ ┃ ┃ ┃ ┃ ┗ 头像图片  
 ┃ ┃ ┃ ┃ ┣ video  
 ┃ ┃ ┃ ┃ ┣ weibo_10.json # 抓取的微博帖文数据，数字表示内含数据条数  
 ┃ ┃ ┃ ┃ ┣ weibo_1997.json  
 ┃ ┃ ┃ ┃ ┗ weibo_400.json  
 ┃ ┃ ┣ weibo_spider # 微博爬虫主要代码   
 ┃ ┃ ┣ requirements.txt  
 ┃ ┃ ┗ setup.py  
 ┃ ┗ xhsSpider  
 ┃ ┃ ┣ datas  
 ┃ ┃ ┃ ┗ json_datas # 抓取到的小红书帖文数据，数字表示内含数据条数  
 ┃ ┃ ┃ ┃ ┣ xhs_10.json  
 ┃ ┃ ┃ ┃ ┣ xhs_400.json  
 ┃ ┃ ┃ ┃ ┗ xhs_543.json  
 ┃ ┃ ┣ xhs_utils  
 ┃ ┃ ┣ apis  
 ┃ ┃ ┣ .env  
 ┃ ┃ ┣ main.py  
 ┃ ┃ ┗ requirements.txt  
 ┣ .env  
 ┣ requirements.txt
 ┣ annotated_risk_dataset.json # 完整数据集
 ┣ semantic_completion_pipeline.py # 语义处理流水线
 ┣ llm_annotation.py # 原型-NLI辅助的LLM标注  
 ┣ data_augmentation.py # 数据增强  
 ┗ model_validation # 模型验证  
```
## 爬虫代码使用

### 微博爬虫
使用https://github.com/dataabc/weiboSpider
支持爬取单个用户某一时间范围内的帖文、图片，输出为txt、json、csv格式  
1.```pip install -r requirements.txt```  
2.weibo_spider/config.json配置微博cookie，需要爬取的用户id和时间范围  
3.```python3 -m weibo_spider```  

### 小红书爬虫
apis与核心签名使用https://github.com/cv-cat/Spider_XHS  
main有修改，支持爬取图文笔记并识别图片中的文字内容  
❗python版本需小于3.11（paddle支持问题）  
1.```pip install -r requirements.txt```  
2.xhsSpider/.env中配置微博cookie  
3.main.py中user_url配置抓取用户主页url  
3.```python main.py```

## 语义补全流水线 semantic_completion_pipeline.py
1.```pip install -r requirements.txt```  
2..env配置openai与deepseek的api  

### 单独运行某个阶段
1.仅筛选与识别暗语，源文件默认weibo_1997与xhs_543，在semantic_completion_pipeline.py中修改  
```
python semantic_completion_pipeline.py --mode filter
```
2.仅更新隐语证据库（必须在筛选后输出的stage1_filtered.json基础上进行）  
```
python semantic_completion_pipeline.py --mode update_slang --input stage1_filtered.json.json
```
3.仅语义补全（必须在筛选后输出的stage1_filtered.json基础上进行） 
```
python semantic_completion_pipeline.py --mode semantic --input stage1_filtered.json.json  
```

### 全流程运行
```
python semantic_completion_pipeline.py --mode full
```

## 原型-NLI辅助LLM标注 llm_annotation.py
input为语义补全阶段输出（final_semantic_completion.json，包含literal_meaning和pragmatic_inference字段） 
建议在cuda环境下运行（否则NLI模型速度会很慢），运行时会自动识别环境 
### 快速启动
```
python llm_annotation.py --input output\20260424_202034\final_semantic_completion.json                           
```
### 参数说明
```--input``` 输出JSON文件路径
```--output``` 输出JSON文件路径（默认: annotated_risk_dataset.json）
```--embedding_model``` LLM模型名称（默认: shibing624/text2vec-base-chinese）
```--llm_model``` LLM模型名称（默认: deepseek-chat）
```--nli_model``` NLI模型名称（默认：MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7）
```--api_key``` 此处输入或在.env里设置
```--base_url``` LLM API Base URL（默认: https://api.deepseek.com）

## 模型验证 model_validation.py
input为标注后数据（默认为annotated_risk_dataset.json），输出output/validation/validation_report_时间戳.json 
### 快速启动 
```
python model_validation.py --input annotated_risk_dataset.json --output_dir output/validation --models tfidf_svm,tfidf_rf,bert,llm_zeroshot --n_folds 1
```
### 参数说明 
```--input``` 数据集文件路径 
```--output_dir``` 输出目录 
```--models``` 要评估的模型，逗号分隔: tfidf_svm,tfidf_rf,bert,llm_zeroshot 
```--n_folds``` 交叉验证折数（默认5）。设为1则退化为单次划分 
```--test_ratio``` 测试集比例（仅n_folds=1时使用，默认0.2） 
```--seed``` 随机种子，默认为42 
```--llm_test_limit``` LLM零样本评估的最大测试样本数（控制API费用） 
```--augment``` 对每折训练集执行数据增强（平方根平衡），需要与data_augmentation.py在同级目录下 
```--evidence``` 暗语证据库路径（增强时使用），默认evidence/slang_emoji_dict.json
```--backtranslation``` 启用回译增强 
```--bert_model``` BERT模型名称或本地路径（默认: bert-base-chinese） 
```--text_mode``` 文本构建模式（消融实验用）: content=仅原文; content_literal=原文+语重写帖文;content_literal_pragmatic=全部字段 （默认） 
```--high_risk_boost``` 等级3/4/5的类别权重额外倍率（默认3.0, 设为1.0则不额外提升）
