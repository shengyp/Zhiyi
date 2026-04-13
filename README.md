## 代码结构
Suicide_Risk  
 ┣ evidence  
 ┃ ┗ slang_emoji_dict.json #暗语证据库  
 ┣ output # 输出数据位置，包含部分测试数据  
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
 ┃ ┃ ┣ weibo_spider   
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
 ┣ archetype_annotation.py # 原型-NLI标注  
 ┣ data_augmentation.py # 数据增强  
 ┣ model_validation # 模型验证  
 ┗ semantic_completion_pipeline.py # 语义补全流水线  

## 爬虫代码使用

### 微博爬虫
使用https://github.com/dataabc/weiboSpider开源项目  
支持爬取单个用户某一时间范围内的帖文、图片，输出为txt、json、csv格式  
1.```pip install -r requirements.txt```
2.weibo_spider/config.json配置微博cookie，需要爬取的用户id和时间范围  
3.```python3 -m weibo_spider```

### 小红书爬虫
❗python版本需小于3.11（paddle支持问题）  
1.pip install -r requirements.txt  
2.xhsSpider/.env中配置微博cookie  
3.main.py中user_url配置抓取用户主页url  
3.```python main.py```

## 语义补全流水线 semantic_completion_pipeline.py
1.pip install -r requirements.txt  
2..env配置openai与deepseek的api  

### 单独运行某个阶段
1.仅筛选与识别暗语，源文件默认weibo_1997与xhs_543，在semantic_completion_pipeline.py中修改  
```
python semantic_completion_pipeline.py --mode filter
```
2.仅更新暗语库（必须在筛选后输出基础上进行）  
```
python semantic_completion_pipeline.py --mode update_slang --input 筛选后_filtered文件.json
```
3.仅语义补全（必须在筛选后输出基础上进行） 
```
python semantic_completion_pipeline.py --mode semantic --input 筛选后_clean文件.json  
```

### 全流程运行
```
python rag_pic_pipeline.py --mode full
```

## 原型-NLI双通道标注 archetype_annotation.py
```
pip install -r requirements.txt
```
### 基本用法
input为语义补全阶段输出（包含literal_meaning和pragmatic_inference字段），calibration为人工标注样本（包含 risk_level 字段（整数 0-5））
```
python archetype_annotation.py --input final_semantic_completion.json --calibration human_labeled_samples.json --output annotated_risk_dataset.json
```


