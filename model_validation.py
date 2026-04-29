"""
支持的模型:
  1. TF-IDF + SVM (传统机器学习基线)
  2. TF-IDF + Random Forest (集成学习基线)
  3. BERT-base-Chinese 微调 (深度学习方法)
  4. 基于LLM的零样本/少样本分类 (DeepSeek zero-shot)

评估方式:
  - 支持 n 折分层交叉验证（每折单独做数据增强，防止数据泄露）
  - 汇报每折结果及 mean ± std

评估指标:
  - 精确率 (Precision), 召回率 (Recall), F1-Score (per-class & macro/weighted)
  - 混淆矩阵
  - Cohen's Kappa
"""

import os
import re
import json
import argparse
import time
import warnings
import importlib.util
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser(description="模型验证 - 自杀风险检测")
parser.add_argument("--input", type=str, required=True,
                    help="数据集文件路径 (annotated_risk_*.json)")
parser.add_argument("--output_dir", type=str, default="output/",
                    help="输出目录")
parser.add_argument("--models", type=str, default="tfidf_svm,tfidf_rf,llm_zeroshot",
                    help="要评估的模型，逗号分隔: tfidf_svm,tfidf_rf,bert,llm_zeroshot")
parser.add_argument("--n_folds", type=int, default=5,
                    help="交叉验证折数 (默认5)。设为1则退化为单次划分")
parser.add_argument("--test_ratio", type=float, default=0.2,
                    help="测试集比例 (仅 n_folds=1 时使用，默认0.2)")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--llm_test_limit", type=int, default=50,
                    help="LLM零样本评估的最大测试样本数（控制API费用）")
# ---- 数据增强参数 ----
parser.add_argument("--augment", action="store_true",
                    help="对每折训练集执行数据增强（平方根平衡）")
parser.add_argument("--evidence", type=str, default="evidence/slang_emoji_dict.json",
                    help="暗语证据库路径（增强时使用）")
parser.add_argument("--backtranslation", action="store_true",
                    help="启用回译增强")
# ---- BERT参数 ----
parser.add_argument("--bert_model", type=str, default="bert-base-chinese",
                    help="BERT模型名称或本地路径 (默认: bert-base-chinese)")
# ---- 消融实验 & 类别权重 ----
parser.add_argument("--text_mode", type=str, default="content_literal_pragmatic",
                    choices=["content", "content_literal", "content_literal_pragmatic"],
                    help="文本构建模式（消融实验用）: "
                         "content=仅原文; "
                         "content_literal=原文+语义改写; "
                         "content_literal_pragmatic=全部字段 (默认)")
parser.add_argument("--high_risk_boost", type=float, default=3.0,
                    help="等级3/4/5的类别权重额外倍率 (默认3.0, 设为1.0则不额外提升)")
args = parser.parse_args()

np.random.seed(args.seed)
load_dotenv()

os.makedirs(args.output_dir, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 工具函数
# ============================================================

def _build_text(item, mode="content_literal_pragmatic"):
    """
    按 text_mode 构建模型输入文本。
    使用 [SEP] 分隔不同语义字段，为 BERT 建立结构感，
    同时对 TF-IDF 模型也能作为段落边界标记。

    模式:
      content                    — 仅原文
      content_literal            — 原文 [SEP] 语义改写
      content_literal_pragmatic  — 原文 [SEP] 语义改写 [SEP] 语用推断
    """
    content = item.get("content", "")
    if mode == "content":
        return content

    literal = item.get("literal_meaning", "")
    parts = [content]
    if literal:
        parts.append(literal)
    if mode == "content_literal":
        return " [SEP] ".join(parts)

    # content_literal_pragmatic
    pragmatic = item.get("pragmatic_inference", "")
    if pragmatic:
        parts.append(pragmatic)
    return " [SEP] ".join(parts)


def _load_augmentation_module():
    """动态加载 data_augmentation.py 模块"""
    aug_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data_augmentation.py"
    )
    if not os.path.exists(aug_script):
        aug_script = "data_augmentation.py"
    spec = importlib.util.spec_from_file_location("data_augmentation", aug_script)
    aug_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aug_module)
    return aug_module


def augment_fold_training_data(train_items, aug_module, fold_seed):
    """
    对单折训练集执行数据增强。

    参数:
      train_items: 原始训练样本列表
      aug_module: 增强模块
      fold_seed: 当前折的随机种子

    返回:
      (augmented_texts, augmented_labels)
    """
    train_counter = Counter(item["risk_level"] for item in train_items)

    augmented_train = aug_module.augmentation_pipeline(
        train_items,
        evidence_path=args.evidence,
        seed=fold_seed,
        enable_backtranslation=args.backtranslation,
    )

    texts = [_build_text(item, mode=args.text_mode) for item in augmented_train]
    labels = np.array([item["risk_level"] for item in augmented_train])
    return texts, labels


# ============================================================
# 类别权重计算（高风险等级加权）
# ============================================================
from sklearn.utils.class_weight import compute_class_weight


def _compute_class_weights(labels, boost=2.0):
    """
    计算类别权重：先用 sklearn balanced 权重作为基础，
    再对等级3/4/5乘以 high_risk_boost 倍率。

    返回:
      weight_dict: {class_label: weight}  — 用于 SVM / RF 的 class_weight 参数
      weight_tensor: list of float        — 按类别顺序排列，用于 BERT 的 CrossEntropyLoss
    """
    classes = np.array(sorted(set(labels)))
    base_weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_dict = {}
    for cls, w in zip(classes, base_weights):
        if cls >= 3:
            weight_dict[int(cls)] = float(w * boost)
        else:
            weight_dict[int(cls)] = float(w)

    # 按 0-5 顺序生成 tensor 用的 list（BERT loss 需要）
    all_classes = list(range(6))
    weight_list = []
    for c in all_classes:
        weight_list.append(weight_dict.get(c, 1.0))

    return weight_dict, weight_list


# ============================================================
# 评估指标函数
# ============================================================
from sklearn.metrics import (classification_report, confusion_matrix,
                              cohen_kappa_score, f1_score)


def evaluate_predictions(model_name, y_true, y_pred, label_names=None, verbose=True):
    """计算模型评估指标，返回结果字典"""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        report = classification_report(y_true, y_pred, zero_division=0,
                                        target_names=label_names)
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")
        print(report)
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Cohen Kappa: {kappa:.4f}")
        print(f"\n混淆矩阵:")
        print(cm)

    return {
        "model": model_name,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# 中文分词预处理
# ============================================================
try:
    import jieba
    def tokenize_zh(text):
        return " ".join(jieba.cut(text))
except ImportError:
    print("警告: jieba未安装，使用字符级分割")
    def tokenize_zh(text):
        return " ".join(list(text))


# ============================================================
# 模型1: TF-IDF + SVM
# ============================================================
def train_tfidf_svm(train_texts, train_labels, test_texts, seed,
                    class_weight_dict=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    train_tok = [tokenize_zh(t) for t in train_texts]
    test_tok = [tokenize_zh(t) for t in test_texts]

    cw = class_weight_dict if class_weight_dict else "balanced"
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("svm", LinearSVC(C=1.0, max_iter=5000, class_weight=cw,
                           random_state=seed)),
    ])

    pipeline.fit(train_tok, train_labels)
    return pipeline.predict(test_tok)


# ============================================================
# 模型2: TF-IDF + Random Forest
# ============================================================
def train_tfidf_rf(train_texts, train_labels, test_texts, seed,
                   class_weight_dict=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    train_tok = [tokenize_zh(t) for t in train_texts]
    test_tok = [tokenize_zh(t) for t in test_texts]

    cw = class_weight_dict if class_weight_dict else "balanced"
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=20,
                                       class_weight=cw,
                                       random_state=seed, n_jobs=-1)),
    ])

    pipeline.fit(train_tok, train_labels)
    return pipeline.predict(test_tok)


# ============================================================
# 模型3: BERT-base-Chinese 微调 (可选)
# ============================================================
def train_bert(train_texts, train_labels, test_texts, seed,
               class_weight_list=None):
    try:
        import torch
        import torch.nn as nn
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   Trainer, TrainingArguments)
    except ImportError:
        print("[BERT] 跳过: 需要安装 transformers 和 torch")
        return None

    model_name = getattr(args, 'bert_model', 'bert-base-chinese')
    tokenizer = None
    model = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_labels = len(set(train_labels))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    except Exception as e1:
        print(f"  HuggingFace加载失败: {e1}")
        try:
            from modelscope import AutoTokenizer as MSTokenizer
            from modelscope import AutoModelForSequenceClassification as MSModel
            print(f"  尝试ModelScope镜像...")
            tokenizer = MSTokenizer.from_pretrained(model_name)
            num_labels = len(set(train_labels))
            model = MSModel.from_pretrained(model_name, num_labels=num_labels)
        except Exception as e2:
            print(f"  ModelScope加载也失败: {e2}")
            print("  [提示] 请手动下载模型到本地，用 --bert_model <本地路径> 指定")
            return None

    class RiskDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tok, max_len=256):
            self.encodings = tok(texts, truncation=True, padding=True,
                                  max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_dataset = RiskDataset(train_texts, train_labels.tolist(), tokenizer)
    test_dataset = RiskDataset(test_texts, [0]*len(test_texts), tokenizer)

    # ---- 自定义 Trainer: 加权交叉熵损失 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if class_weight_list is not None:
        loss_weights = torch.tensor(class_weight_list, dtype=torch.float32).to(device)
        print(f"  [BERT] 类别权重: {[f'{w:.2f}' for w in class_weight_list]}")
    else:
        loss_weights = None

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if loss_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "bert_checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        seed=seed,
        use_cpu=not torch.cuda.is_available(),
    )

    model.to(device)
    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=train_dataset
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    return np.argmax(predictions.predictions, axis=-1)


# ============================================================
# 模型4: LLM零样本分类 (DeepSeek)
# ============================================================
def eval_llm_zeroshot(test_texts, test_labels, limit):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    n = min(len(test_texts), limit)
    indices = np.random.choice(len(test_texts), n, replace=False)
    subset_texts = [test_texts[i] for i in indices]
    subset_labels = test_labels[indices]

    rubric_text = """等级0 - 无自杀意念: 文本不包含任何与自杀、自伤相关的意念或暗示。
等级1 - 希望死亡但无计划: 表达了希望死亡、不想活等消极意愿，但未提及具体方法。
等级2 - 有自杀想法但无具体计划: 出现了关于自杀的想法或暗示，但未形成具体计划。
等级3 - 有自杀计划但未实施: 出现了具体的自杀计划元素，如特定方法、时间安排或准备行为。
等级4 - 有自杀未遂或自伤行为: 描述了实际的自伤行为或自杀未遂经历。
等级5 - 高致死性自杀企图: 描述了高致死性的自杀企图或迫在眉睫的自杀意图。"""

    preds = []
    for i, text in enumerate(tqdm(subset_texts, desc="LLM零样本预测")):
        prompt = f"""请根据以下C-SSRS衍生的风险等级标准，判断给定文本的自杀风险等级(0-5)。
注意中文社交媒体中可能存在暗语和隐含表达。

## 等级标准:
{rubric_text}

## 待评估文本:
{text}

请只输出一个0-5之间的整数，不需要任何解释。"""

        time.sleep(0.3)
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是自杀风险评估专家，只输出0-5的整数。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=5,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r'[0-5]', raw)
            preds.append(int(match.group()) if match else 0)
        except Exception as e:
            print(f"  LLM预测失败 ({i}): {e}")
            preds.append(0)

    return np.array(preds), subset_labels


# ============================================================
# 主流程
# ============================================================
def main():
    model_list = [m.strip() for m in args.models.split(",")]
    label_names = [f"等级{i}" for i in range(6)]

    # ============================================================
    # 数据加载
    # ============================================================
    print(f"加载数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict) and "data" in raw_data:
        raw_data = raw_data["data"]

    data = [item for item in raw_data if item.get("risk_level", -1) >= 0]
    print(f"有效数据: {len(data)} 条 (过滤掉 {len(raw_data) - len(data)} 条标注失败数据)")

    # 分离原始/合成样本（测试集只用原始样本，防止数据泄露）
    original_data = [item for item in data if not item.get("is_synthetic", False)]
    synthetic_data = [item for item in data if item.get("is_synthetic", False)]
    print(f"原始样本: {len(original_data)} 条, 合成样本: {len(synthetic_data)} 条")

    orig_labels = np.array([item["risk_level"] for item in original_data])
    print(f"\n原始样本标签分布:")
    for lv, cnt in sorted(Counter(orig_labels).items()):
        print(f"  等级 {lv}: {cnt} 条 ({cnt/len(orig_labels)*100:.1f}%)")

    # 加载增强模块（如果需要）
    aug_module = None
    if args.augment:
        try:
            aug_module = _load_augmentation_module()
            print(f"\n[增强] 已加载增强模块，策略: 平方根平衡 (sqrt)")
        except Exception as e:
            print(f"[增强] 加载增强模块失败: {e}，将使用未增强数据")

    # ============================================================
    # 交叉验证设置
    # ============================================================
    n_folds = args.n_folds
    print(f"\n评估方式: {n_folds}折分层交叉验证" if n_folds > 1
          else f"\n评估方式: 单次划分 (测试集比例={args.test_ratio})")

    if n_folds > 1:
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=args.seed)
        fold_splits = list(kfold.split(original_data, orig_labels))
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio,
                                          random_state=args.seed)
        fold_splits = list(splitter.split(original_data, orig_labels))

    # ============================================================
    # 收集每折每个模型的结果
    # ============================================================
    # model_name -> list of metric dicts (one per fold)
    fold_results = defaultdict(list)

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        print(f"\n{'#'*60}")
        print(f"  第 {fold_num}/{len(fold_splits)} 折")
        print(f"{'#'*60}")

        # ----- 构建当前折的训练/测试集 -----
        test_items = [original_data[i] for i in test_idx]
        test_texts = [_build_text(item, mode=args.text_mode) for item in test_items]
        test_labels = np.array([item["risk_level"] for item in test_items])

        train_original_items = [original_data[i] for i in train_idx]

        # ----- 数据增强（仅对训练集） -----
        if aug_module is not None:
            fold_seed = args.seed + fold_idx
            print(f"[增强] 对第{fold_num}折训练集执行增强 "
                  f"(策略=sqrt, seed={fold_seed})")
            try:
                train_texts, train_labels = augment_fold_training_data(
                    train_original_items, aug_module, fold_seed
                )
                print(f"[增强] 增强后训练集: {len(train_texts)} 条, "
                      f"分布: {dict(sorted(Counter(train_labels).items()))}")
            except Exception as e:
                print(f"[增强] 第{fold_num}折增强失败: {e}，使用原始训练集")
                train_texts = [_build_text(item, mode=args.text_mode)
                              for item in train_original_items]
                train_labels = np.array([item["risk_level"]
                                         for item in train_original_items])
        else:
            train_texts = [_build_text(item, mode=args.text_mode)
                          for item in train_original_items]
            train_labels = np.array([item["risk_level"]
                                     for item in train_original_items])
            # 如果有预有合成样本且未启用 --augment，加入训练集
            if synthetic_data:
                for item in synthetic_data:
                    train_texts.append(_build_text(item, mode=args.text_mode))
                train_labels = np.concatenate([
                    train_labels,
                    np.array([item["risk_level"] for item in synthetic_data])
                ])

        print(f"训练集: {len(train_texts)} 条, 测试集: {len(test_texts)} 条")
        print(f"训练集分布: {dict(sorted(Counter(train_labels).items()))}")
        print(f"测试集分布: {dict(sorted(Counter(test_labels).items()))}")
        print(f"文本模式: {args.text_mode}")

        # ----- 计算类别权重 -----
        cw_dict, cw_list = _compute_class_weights(
            train_labels, boost=args.high_risk_boost
        )
        print(f"类别权重 (boost={args.high_risk_boost}x for L3/4/5): "
              f"{{{', '.join(f'{k}:{v:.2f}' for k, v in sorted(cw_dict.items()))}}}")

        # ----- 逐模型训练与评估 -----
        for model_key in model_list:
            verbose = (len(fold_splits) == 1)  # 单折时打印详细报告
            try:
                if model_key == "tfidf_svm":
                    preds = train_tfidf_svm(train_texts, train_labels,
                                             test_texts, args.seed,
                                             class_weight_dict=cw_dict)
                    result = evaluate_predictions(
                        "TF-IDF + SVM", test_labels, preds,
                        label_names, verbose=verbose)

                elif model_key == "tfidf_rf":
                    preds = train_tfidf_rf(train_texts, train_labels,
                                            test_texts, args.seed,
                                            class_weight_dict=cw_dict)
                    result = evaluate_predictions(
                        "TF-IDF + Random Forest", test_labels, preds,
                        label_names, verbose=verbose)

                elif model_key == "bert":
                    preds = train_bert(train_texts, train_labels,
                                        test_texts, args.seed,
                                        class_weight_list=cw_list)
                    if preds is not None:
                        result = evaluate_predictions(
                            "BERT-base-Chinese", test_labels, preds,
                            label_names, verbose=verbose)
                    else:
                        result = {"model": "BERT-base-Chinese", "error": "依赖未安装"}

                elif model_key == "llm_zeroshot":
                    # LLM零样本太贵，只在第1折评估
                    if fold_idx > 0:
                        print(f"  [LLM Zero-Shot] 仅在第1折评估，跳过第{fold_num}折")
                        continue
                    preds, subset_labels = eval_llm_zeroshot(
                        test_texts, test_labels, args.llm_test_limit)
                    result = evaluate_predictions(
                        "DeepSeek Zero-Shot", subset_labels, preds,
                        label_names, verbose=True)

                else:
                    print(f"未知模型: {model_key}，跳过")
                    continue

                if "error" not in result:
                    print(f"  [{result['model']}] 第{fold_num}折: "
                          f"Macro F1={result['macro_f1']:.4f}, "
                          f"Weighted F1={result['weighted_f1']:.4f}, "
                          f"Kappa={result['cohen_kappa']:.4f}")
                fold_results[model_key].append(result)

            except Exception as e:
                print(f"模型 {model_key} 第{fold_num}折评估异常: {e}")
                fold_results[model_key].append({"model": model_key, "error": str(e)})

    # ============================================================
    # 汇总交叉验证结果
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  交叉验证汇总 ({len(fold_splits)} 折)")
    print(f"{'='*70}")

    summary = []
    for model_key in model_list:
        results = fold_results.get(model_key, [])
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            model_display = results[0]["model"] if results else model_key
            print(f"\n{model_display}: 所有折均失败")
            summary.append({"model": model_display, "error": "所有折均失败"})
            continue

        model_display = valid_results[0]["model"]
        macro_f1s = [r["macro_f1"] for r in valid_results]
        weighted_f1s = [r["weighted_f1"] for r in valid_results]
        kappas = [r["cohen_kappa"] for r in valid_results]

        if len(valid_results) > 1:
            print(f"\n{model_display}  ({len(valid_results)} 折)")
            print(f"  Macro F1:    {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}  "
                  f"(各折: {', '.join(f'{v:.4f}' for v in macro_f1s)})")
            print(f"  Weighted F1: {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}  "
                  f"(各折: {', '.join(f'{v:.4f}' for v in weighted_f1s)})")
            print(f"  Cohen Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}  "
                  f"(各折: {', '.join(f'{v:.4f}' for v in kappas)})")
        else:
            print(f"\n{model_display}")
            print(f"  Macro F1:    {macro_f1s[0]:.4f}")
            print(f"  Weighted F1: {weighted_f1s[0]:.4f}")
            print(f"  Cohen Kappa: {kappas[0]:.4f}")

        summary.append({
            "model": model_display,
            "n_folds": len(valid_results),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "weighted_f1_mean": float(np.mean(weighted_f1s)),
            "weighted_f1_std": float(np.std(weighted_f1s)),
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas)),
            "per_fold": valid_results,
        })

    # ---- 横向对比表格 ----
    print(f"\n{'='*70}")
    print(f"{'模型':<25} {'Macro F1':>15} {'Weighted F1':>15} {'Kappa':>15}")
    print("-" * 70)
    for s in summary:
        if "error" in s:
            print(f"{s['model']:<25} {'ERROR':>15}")
        elif s.get("n_folds", 1) > 1:
            print(f"{s['model']:<25} "
                  f"{s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f}  "
                  f"{s['weighted_f1_mean']:.4f}±{s['weighted_f1_std']:.4f}  "
                  f"{s['kappa_mean']:.4f}±{s['kappa_std']:.4f}")
        else:
            print(f"{s['model']:<25} "
                  f"{s['macro_f1_mean']:>15.4f} "
                  f"{s['weighted_f1_mean']:>15.4f} "
                  f"{s['kappa_mean']:>15.4f}")

    # ---- 保存报告 ----
    report = {
        "run_id": run_id,
        "n_folds": len(fold_splits),
        "text_mode": args.text_mode,
        "high_risk_boost": args.high_risk_boost,
        "augment": args.augment,
        "balance_strategy": "sqrt" if args.augment else None,
        "seed": args.seed,
        "results": summary,
    }
    report_path = os.path.join(args.output_dir, f"validation_report_{run_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n评估报告保存至: {report_path}")


if __name__ == "__main__":
    main()
