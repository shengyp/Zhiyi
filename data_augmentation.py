"""
增强策略：
  1. 暗语互换：证据库中同义暗语互相替换（紫砂↔zs）
  2. 语义字段复用：literal_meaning作为独立样本、pragmatic拼接
  3. 离线回译：对literal_meaning做 中→英→中 回译产生自然变体
  4. 社交媒体噪声注入：插入语气词/标点/重复/截断（不改变任何实词）
  5. 句子级操作：多句文本的句序打乱、句子丢弃
  6. 随机过采样补齐

依赖:
  pip install jieba numpy tqdm
  # 可选(回译): pip install argostranslate
"""
import os
import sys
import json
import math
import random
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
from tqdm import tqdm

# ============================================================================
# 依赖检查
# ============================================================================

def _check_dependencies():
    missing = []
    for pkg, import_name in [
        ("jieba", "jieba"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] 缺少依赖包: {', '.join(missing)}")
        print(f"  请运行: pip install {' '.join(missing)}")
        sys.exit(1)

_check_dependencies()

import jieba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 动态保护词构建器（与v2相同，从证据库动态获取）
# ============================================================================

class DynamicProtectedWords:
    """
    从暗语证据库和样本的evidence_used字段动态构建保护词集合。
    不依赖硬编码列表，而是根据实际数据中出现的暗语自动适配。
    """

    def __init__(self, evidence_path: Optional[str] = None):
        self.global_slang_set: Set[str] = set()
        self.slang_to_meaning: Dict[str, str] = {}
        self.meaning_to_slangs: Dict[str, List[str]] = defaultdict(list)

        if evidence_path and os.path.exists(evidence_path):
            self._load_evidence_db(evidence_path)
            logger.info(f"[保护词] 从证据库加载 {len(self.global_slang_set)} 个暗语")

    def _load_evidence_db(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            evidence_data = json.load(f)
        for item in evidence_data:
            surface = item.get("surface", "").strip()
            meaning = item.get("meaning", "").strip()
            if surface:
                self.global_slang_set.add(surface)
                if meaning:
                    self.slang_to_meaning[surface] = meaning
                    self.meaning_to_slangs[meaning].append(surface)

    def get_protected_set(self, item: Dict[str, Any]) -> Set[str]:
        """获取单条样本的保护词集合（evidence_used + 全局暗语库子串匹配）"""
        protected = set()
        ev_used = item.get("evidence_used", [])
        if isinstance(ev_used, list):
            for ev in ev_used:
                ev_clean = str(ev).strip()
                if ev_clean:
                    protected.add(ev_clean)
        content = item.get("content", "")
        for slang in self.global_slang_set:
            if slang in content:
                protected.add(slang)
        return protected

    def get_similar_slangs(self, slang: str) -> List[str]:
        """获取与给定暗语语义相同的其他暗语（基于证据库meaning字段）"""
        meaning = self.slang_to_meaning.get(slang, "")
        if not meaning:
            return []
        return [s for s in self.meaning_to_slangs.get(meaning, []) if s != slang]


# ============================================================================
# 2. 辅助函数
# ============================================================================

def _tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


def _tokens_to_text(tokens: List[str]) -> str:
    return "".join(tokens)


def _is_in_protected(word: str, protected_set: Set[str]) -> bool:
    """双向子串包含判断：暗语在token中 或 token在暗语中"""
    w = word.strip()
    if not w:
        return False
    for p in protected_set:
        if p in w or w in p:
            return True
    return False


def _split_sentences(text: str) -> List[str]:
    """按中文标点和换行符分割句子"""
    parts = re.split(r'([。！？!?\n]+)', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    return [s for s in sentences if s.strip()]


# ============================================================================
# 3. 暗语替换增强器（从证据库中同义暗语互换，与v2相同）
# ============================================================================

class SlangSubstitutionAugmenter:
    """
    利用证据库中语义相同的暗语进行互相替换。
    例如"紫砂"和"zs"的meaning都是"自杀"，
    那么"我想紫砂"可以生成变体"我想zs"。
    """

    def __init__(self, protector: DynamicProtectedWords):
        self.protector = protector

    def augment(self, item: Dict[str, Any], n_aug: int = 2) -> List[Dict[str, Any]]:
        content = item.get("content", "")
        evidence_used = item.get("evidence_used", [])
        if not isinstance(evidence_used, list):
            evidence_used = []

        replaceable_slangs = []
        for slang in evidence_used:
            slang = str(slang).strip()
            if slang and slang in content:
                alternatives = self.protector.get_similar_slangs(slang)
                if alternatives:
                    replaceable_slangs.append((slang, alternatives))

        for slang in self.protector.global_slang_set:
            if slang in content and slang not in [s for s, _ in replaceable_slangs]:
                alternatives = self.protector.get_similar_slangs(slang)
                if alternatives:
                    replaceable_slangs.append((slang, alternatives))

        if not replaceable_slangs:
            return []

        results = []
        for _ in range(n_aug):
            new_content = content
            new_evidence = list(evidence_used)
            n_replace = min(len(replaceable_slangs), random.randint(1, 2))
            selected = random.sample(replaceable_slangs, n_replace)

            for old_slang, alternatives in selected:
                new_slang = random.choice(alternatives)
                new_content = new_content.replace(old_slang, new_slang, 1)
                new_evidence = [new_slang if e == old_slang else e
                                for e in new_evidence]

            if new_content != content:
                new_item = deepcopy(item)
                new_item["content"] = new_content
                new_item["evidence_used"] = new_evidence
                new_item["is_synthetic"] = True
                new_item["augmentation_method"] = "slang_substitution"
                results.append(new_item)

        return results


# ============================================================================
# 4. 语义字段复用增强器
# ============================================================================

class SemanticFieldAugmenter:
    """
    利用RAG阶段已生成的语义补全字段进行数据增强，不做任何词义替换。

    策略a: literal_meaning作为独立样本
    策略b: 原文前半段 + literal_meaning后半段
    策略c: 原文 + pragmatic_inference追加
    策略d: 同等级样本间的语义交叉
    """

    def augment_single(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = item.get("content", "").strip()
        literal = item.get("literal_meaning", "").strip()
        pragmatic = item.get("pragmatic_inference", "").strip()
        results = []

        # 策略a: literal_meaning作为独立样本
        if literal and literal != content and len(literal) >= 10:
            new_item = deepcopy(item)
            new_item["content"] = literal
            new_item["is_synthetic"] = True
            new_item["augmentation_method"] = "literal_as_sample"
            new_item["evidence_used"] = []
            results.append(new_item)

        # 策略b: 原文前半段 + literal后半段
        if literal and content and literal != content:
            content_half = content[:len(content) // 2]
            literal_half = literal[len(literal) // 2:]
            if content_half and literal_half:
                combined = content_half + literal_half
                new_item = deepcopy(item)
                new_item["content"] = combined
                new_item["is_synthetic"] = True
                new_item["augmentation_method"] = "content_literal_splice"
                results.append(new_item)

        # 策略c: 原文 + pragmatic_inference追加
        if pragmatic and len(pragmatic) >= 15:
            enriched = f"{content} {pragmatic}"
            new_item = deepcopy(item)
            new_item["content"] = enriched
            new_item["is_synthetic"] = True
            new_item["augmentation_method"] = "pragmatic_enriched"
            results.append(new_item)

        return results

    def cross_augment(self, items_same_class: List[Dict[str, Any]],
                      n_aug: int = 2) -> List[Dict[str, Any]]:
        """同等级样本之间的语义交叉增强（A原文 + B的pragmatic）"""
        results = []
        if len(items_same_class) < 2:
            return results

        pragmatics = [
            (i, item.get("pragmatic_inference", "").strip())
            for i, item in enumerate(items_same_class)
            if item.get("pragmatic_inference", "").strip()
        ]
        if len(pragmatics) < 2:
            return results

        for _ in range(min(n_aug, len(items_same_class))):
            idx_a, idx_b = random.sample(range(len(pragmatics)), 2)
            item_a_idx, _ = pragmatics[idx_a]
            _, prag_b = pragmatics[idx_b]
            item_a = items_same_class[item_a_idx]

            combined = f"{item_a.get('content', '')} {prag_b}"
            new_item = deepcopy(item_a)
            new_item["content"] = combined
            new_item["is_synthetic"] = True
            new_item["augmentation_method"] = "cross_pragmatic"
            results.append(new_item)

        return results


# ============================================================================
# 5. 离线回译增强器（argostranslate zh→en→zh）
# ============================================================================

class BackTranslationAugmenter:
    """
    对literal_meaning做离线回译（中→英→中）产生自然变体。

    为什么用literal_meaning而不是原文content？
    - content含暗语（"紫砂"），翻译引擎不认识暗语，会产生乱译
    - literal_meaning是去暗语后的直白中文（"自杀"），翻译质量更高
    - 回译后的文本仍然表达相同的风险含义，但遣词造句自然不同

    依赖: pip install argostranslate
    首次运行会自动下载中英翻译模型（约100MB）。
    如果环境没有argostranslate，该增强器会静默跳过，不影响其他策略。
    """

    def __init__(self):
        self.available = False
        self._translate_zh_en = None
        self._translate_en_zh = None
        try:
            import argostranslate.package
            import argostranslate.translate
            self._ensure_models()
            self.available = True
            logger.info("[回译] argostranslate 可用，已启用回译增强")
        except ImportError:
            logger.info("[回译] argostranslate 未安装，跳过回译增强"
                        "（可选安装: pip install argostranslate）")
        except Exception as e:
            logger.warning(f"[回译] argostranslate 初始化失败: {e}")

    def _ensure_models(self):
        """确保中英双向翻译模型已下载"""
        import argostranslate.package
        import argostranslate.translate

        installed = argostranslate.translate.get_installed_languages()
        lang_codes = {lang.code for lang in installed}

        if "zh" not in lang_codes or "en" not in lang_codes:
            logger.info("[回译] 正在下载翻译模型...")
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            for pkg in available:
                if (pkg.from_code == "zh" and pkg.to_code == "en") or \
                   (pkg.from_code == "en" and pkg.to_code == "zh"):
                    argostranslate.package.install_from_path(pkg.download())

        installed = argostranslate.translate.get_installed_languages()
        lang_map = {lang.code: lang for lang in installed}
        zh = lang_map.get("zh")
        en = lang_map.get("en")
        if zh and en:
            self._translate_zh_en = zh.get_translation(en)
            self._translate_en_zh = en.get_translation(zh)
        else:
            raise RuntimeError("无法加载中英翻译模型")

    def back_translate(self, text: str) -> Optional[str]:
        """中→英→中回译"""
        if not self.available or not text.strip():
            return None
        try:
            en_text = self._translate_zh_en.translate(text)
            zh_back = self._translate_en_zh.translate(en_text)
            # 回译结果和原文完全相同则无增强效果
            if zh_back.strip() == text.strip():
                return None
            return zh_back.strip()
        except Exception:
            return None

    def augment(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对literal_meaning做回译，生成变体样本"""
        if not self.available:
            return []

        literal = item.get("literal_meaning", "").strip()
        if not literal or len(literal) < 10:
            return []

        results = []
        bt_text = self.back_translate(literal)
        if bt_text and len(bt_text) >= 8:
            new_item = deepcopy(item)
            new_item["content"] = bt_text
            new_item["is_synthetic"] = True
            new_item["augmentation_method"] = "back_translation"
            new_item["evidence_used"] = []  # 回译后暗语已消失
            results.append(new_item)

        return results


# ============================================================================
# 6. 社交媒体噪声注入器
# ============================================================================

class SocialMediaNoiseInjector:
    """
    模拟社交媒体中常见的文本噪声，在不改变任何实词含义的前提下增加多样性。

    四种噪声类型：
    a) 语气词插入：在非保护词之间随机插入语气词（啊、呢、吧、嘛、哦、唉、嗯）
    b) 标点变异：句末标点替换为社交风格（。→... 、→~、！→!!）
    c) 字符重复：随机选一个非保护的汉字重复1-2次（"累"→"累累累"）
    d) 尾部截断：随机去掉文本最后几个非关键字符（模拟发送时截断）

    为什么这些操作是安全的？
    - 没有任何操作改变实词的词义
    - 语气词不携带语义信息，只改变语气
    - 标点变异不改变文字内容
    - 字符重复在社交媒体中很常见（"好累累累"）
    - 截断只去掉尾部，暗语通常在中段或前段
    """

    # 社交媒体常见语气词
    MOOD_PARTICLES = ["啊", "呢", "吧", "嘛", "哦", "唉", "嗯", "呀",
                      "哈", "嘿", "诶", "哎", "额"]

    # 标点变异映射
    PUNCT_VARIANTS = {
        "。": ["...", "。。。", "~"],
        "！": ["!!", "!!!", "！！"],
        "？": ["??", "???", "？？"],
        "，": ["，，", "~", " "],
        "…": ["......", "....", ".."],
    }

    def __init__(self, noise_prob: float = 0.3):
        """noise_prob: 每个可操作位置的噪声注入概率"""
        self.noise_prob = noise_prob

    def _insert_mood_particles(self, text: str, protected: Set[str]) -> str:
        """在非保护区域随机插入语气词"""
        tokens = _tokenize(text)
        new_tokens = []
        for i, tok in enumerate(tokens):
            new_tokens.append(tok)
            if (not _is_in_protected(tok, protected)
                    and len(tok.strip()) > 0
                    and random.random() < self.noise_prob * 0.5):
                new_tokens.append(random.choice(self.MOOD_PARTICLES))
        return _tokens_to_text(new_tokens)

    def _punctuation_variant(self, text: str) -> str:
        """替换句末标点为社交风格变体"""
        result = list(text)
        for i, ch in enumerate(result):
            if ch in self.PUNCT_VARIANTS and random.random() < self.noise_prob:
                result[i] = random.choice(self.PUNCT_VARIANTS[ch])
        return "".join(result)

    def _char_repetition(self, text: str, protected: Set[str]) -> str:
        """随机重复一个非保护区域的汉字"""
        tokens = _tokenize(text)
        candidates = [
            i for i, tok in enumerate(tokens)
            if not _is_in_protected(tok, protected)
            and len(tok.strip()) == 1
            and '\u4e00' <= tok <= '\u9fff'
        ]
        if not candidates:
            return text
        idx = random.choice(candidates)
        repeat_n = random.randint(2, 3)
        tokens[idx] = tokens[idx] * repeat_n
        return _tokens_to_text(tokens)

    def _tail_truncation(self, text: str, protected: Set[str]) -> str:
        """去掉尾部几个非关键字符（最多去掉20%）"""
        max_cut = max(1, len(text) // 5)
        cut_len = random.randint(1, max_cut)
        truncated = text[:-cut_len].rstrip()
        # 确保没有截断到保护词
        for p in protected:
            if p in text and p not in truncated:
                return text  # 保护词被截断了，放弃截断
        if len(truncated) < 5:
            return text
        return truncated

    def augment(self, text: str, protected: Set[str], n_aug: int = 2) -> List[str]:
        """对一条文本生成n_aug个噪声变体"""
        if len(text.strip()) < 5:
            return []

        ops = [
            lambda t: self._insert_mood_particles(t, protected),
            lambda t: self._punctuation_variant(t),
            lambda t: self._char_repetition(t, protected),
            lambda t: self._tail_truncation(t, protected),
        ]

        results = []
        for _ in range(n_aug):
            # 每次随机组合1-2种噪声操作
            n_ops = random.randint(1, 2)
            selected_ops = random.sample(ops, n_ops)
            aug_text = text
            for op in selected_ops:
                aug_text = op(aug_text)
            if aug_text != text and len(aug_text.strip()) >= 5:
                results.append(aug_text)

        return results


# ============================================================================
# 7. 句子级操作增强器
# ============================================================================

class SentenceLevelAugmenter:
    """
    句子级别的文本增强操作。只对含多句的文本有效。

    操作类型：
    a) 句序打乱：随机重排句子顺序（保留含暗语的句子位置不变）
    b) 句子丢弃：随机去掉一句非关键句（不含暗语的句子）

    为什么句子级操作是安全的？
    - 社交媒体帖子的各句之间往往是并列关系（"好累。不想活了。每天都加班。"），
      打乱顺序不改变整体语义
    - 丢弃非关键句只减少背景信息，不改变风险信号
    - 含暗语的句子永远不动、不删
    """

    def augment(self, text: str, protected: Set[str], n_aug: int = 2) -> List[str]:
        sentences = _split_sentences(text)
        if len(sentences) < 3:
            return []  # 少于3句没有操作空间

        # 标记哪些句子含保护词
        protected_indices = set()
        for i, sent in enumerate(sentences):
            for p in protected:
                if p in sent:
                    protected_indices.add(i)
                    break

        # 如果所有句子都含保护词，没有操作空间
        modifiable = [i for i in range(len(sentences)) if i not in protected_indices]
        if not modifiable:
            return []

        results = []
        for _ in range(n_aug):
            op = random.choice(["shuffle", "drop"])

            if op == "shuffle" and len(sentences) >= 3:
                # 只打乱非保护句子的相对顺序
                new_sents = list(sentences)
                mod_sents = [new_sents[i] for i in modifiable]
                random.shuffle(mod_sents)
                for j, idx in enumerate(modifiable):
                    new_sents[idx] = mod_sents[j]
                aug_text = "".join(new_sents)

            elif op == "drop" and len(modifiable) >= 1:
                # 随机丢弃一句非保护句
                drop_idx = random.choice(modifiable)
                new_sents = [s for i, s in enumerate(sentences) if i != drop_idx]
                aug_text = "".join(new_sents)

            else:
                continue

            if aug_text != text and len(aug_text.strip()) >= 5:
                results.append(aug_text)

        return results


# ============================================================================
# 8. 分层平衡策略：计算每类目标样本数
# ============================================================================

def compute_class_targets(
    class_counts: Dict[int, int],
    strategy: str = "sqrt",
    beta: float = 0.99,
    custom_target: Optional[int] = None,
) -> Dict[int, int]:
    """
    根据平衡策略计算每个类别的目标样本数。

    参数:
      class_counts: {类别: 原始样本数}
      strategy: 平衡策略名称
      beta: 有效样本数策略的衰减参数 (仅 effective_number 使用)
      custom_target: 自定义统一目标数 (仅 equal 策略使用)

    策略:
      - "equal":
          所有类别统一增强到同一数量 (当前默认行为)。
          target_i = custom_target 或 max(n_i)

      - "sqrt" (推荐):
          平方根平衡 (Mahajan et al., ECCV 2018)。
          在完全平衡和原始分布之间取几何平均。
          target_i = ceil(sqrt(n_i × n_max))
          少数类大幅增强，但不会增强到与多数类一样多，
          保留了一部分自然分布的先验信息。

      - "log":
          对数平衡。比sqrt更温和，适合极端不平衡时使用。
          target_i = ceil(n_max × log(1 + n_i) / log(1 + n_max))
          少数类增强幅度小于sqrt，更保守。

      - "effective_number":
          有效样本数平衡 (Cui et al., CVPR 2019)。
          E(n) = (1 − β^n) / (1 − β)
          target_i = ceil(n_i × E(n_max) / E(n_i))
          β 越大（接近1）越接近完全平衡，β 越小越接近原始分布。
          推荐: β=0.99 (中等平衡), β=0.999 (接近完全平衡), β=0.9 (温和平衡)

    返回:
      {类别: 目标样本数}，保证 target >= 原始数量
    """
    n_max = max(class_counts.values())

    if strategy == "equal":
        target = custom_target if custom_target else n_max
        return {cls: target for cls in class_counts}

    elif strategy == "sqrt":
        targets = {}
        for cls, n in class_counts.items():
            targets[cls] = max(n, math.ceil(math.sqrt(n * n_max)))
        return targets

    elif strategy == "log":
        targets = {}
        log_max = math.log(1 + n_max)
        for cls, n in class_counts.items():
            log_n = math.log(1 + n)
            targets[cls] = max(n, math.ceil(n_max * log_n / log_max))
        return targets

    elif strategy == "effective_number":
        def _effective_num(n, b):
            if b <= 0:
                return 1.0
            if b >= 1.0:
                return float(n)
            return (1.0 - b ** n) / (1.0 - b)

        e_max = _effective_num(n_max, beta)
        targets = {}
        for cls, n in class_counts.items():
            e_n = _effective_num(n, beta)
            targets[cls] = max(n, math.ceil(n * e_max / e_n))
        return targets

    else:
        raise ValueError(f"未知的平衡策略: {strategy}。"
                         f"可选: equal, sqrt, log, effective_number")


# ============================================================================
# 9. 组合增强管线
# ============================================================================

def augmentation_pipeline(
    data: List[Dict[str, Any]],
    evidence_path: Optional[str] = None,
    target_count: Optional[int] = None,
    seed: int = 42,
    enable_backtranslation: bool = False,
    balance_strategy: str = "sqrt",
    balance_beta: float = 0.99,
) -> List[Dict[str, Any]]:
    """
    无同义词表的暗语感知组合增强管线。

    参数:
      data: 原始数据列表
      evidence_path: 暗语证据库路径
      target_count: 自定义目标数 (仅 equal 策略使用；其他策略自动计算)
      seed: 随机种子
      enable_backtranslation: 是否启用离线回译
      balance_strategy: 平衡策略 ("equal", "sqrt", "log", "effective_number")
      balance_beta: 有效样本数策略的β参数 (仅 effective_number 使用)

    执行顺序（优先级递减）:
    1. 暗语互换（证据库中同义暗语替换，如紫砂↔zs）
    2. 语义字段复用（literal_meaning独立样本、pragmatic拼接等）
    3. 离线回译（对literal_meaning做中→英→中，可选）
    4. 社交媒体噪声注入（语气词/标点/重复/截断）
    5. 句子级操作（句序打乱、句子丢弃）
    6. 随机过采样补齐
    """
    random.seed(seed)
    np.random.seed(seed)

    # 初始化组件
    protector = DynamicProtectedWords(evidence_path)
    slang_aug = SlangSubstitutionAugmenter(protector)
    semantic_aug = SemanticFieldAugmenter()
    bt_aug = BackTranslationAugmenter() if enable_backtranslation else None
    noise_aug = SocialMediaNoiseInjector(noise_prob=0.3)
    sentence_aug = SentenceLevelAugmenter()

    # 按类别分组
    class_groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in data:
        class_groups[item["risk_level"]].append(item)

    counter = {k: len(v) for k, v in sorted(class_groups.items())}

    # 计算每类目标样本数（分层平衡）
    class_targets = compute_class_targets(
        counter,
        strategy=balance_strategy,
        beta=balance_beta,
        custom_target=target_count,
    )

    logger.info(f"[管线] 原始分布: {counter}")
    logger.info(f"[管线] 平衡策略: {balance_strategy}"
                + (f" (β={balance_beta})" if balance_strategy == "effective_number" else ""))
    logger.info(f"[管线] 各类目标: {dict(sorted(class_targets.items()))}")
    logger.info(f"[管线] 全局暗语库: {len(protector.global_slang_set)} 个暗语")
    if bt_aug and bt_aug.available:
        logger.info("[管线] 回译增强: 已启用")
    else:
        logger.info("[管线] 回译增强: 未启用")

    augmented_data = list(data)  # 保留所有原始样本

    for cls in sorted(class_groups.keys()):
        cls_data = class_groups[cls]
        n_current = len(cls_data)
        cls_target = class_targets[cls]
        if n_current >= cls_target:
            logger.info(f"  Level {cls}: {n_current} >= {cls_target}, 跳过")
            continue

        n_needed = cls_target - n_current
        logger.info(f"  Level {cls}: {n_current} -> 目标 {cls_target}, 需生成 {n_needed} 条")

        seen_texts = set()
        new_items = []

        def _try_add(item: Dict[str, Any]) -> bool:
            """尝试添加一条增强样本，返回是否成功"""
            if len(new_items) >= n_needed:
                return False
            c = item.get("content", "")
            if c in seen_texts or len(c.strip()) < 5:
                return False
            seen_texts.add(c)
            new_items.append(item)
            return True

        # === 阶段1: 暗语互换 ===
        for item in cls_data:
            if len(new_items) >= n_needed:
                break
            for r in slang_aug.augment(item, n_aug=2):
                _try_add(r)

        n_stage1 = len(new_items)
        logger.info(f"    [1] 暗语互换: +{n_stage1} 条")

        # === 阶段2: 语义字段复用 ===
        for item in cls_data:
            if len(new_items) >= n_needed:
                break
            for r in semantic_aug.augment_single(item):
                _try_add(r)

        if len(new_items) < n_needed:
            cross_results = semantic_aug.cross_augment(
                cls_data, n_aug=min(n_needed - len(new_items), len(cls_data))
            )
            for r in cross_results:
                _try_add(r)

        n_stage2 = len(new_items) - n_stage1
        logger.info(f"    [2] 语义字段复用: +{n_stage2} 条")

        # === 阶段3: 离线回译 ===
        n_before_bt = len(new_items)
        if bt_aug and bt_aug.available:
            for item in cls_data:
                if len(new_items) >= n_needed:
                    break
                for r in bt_aug.augment(item):
                    _try_add(r)

        n_stage3 = len(new_items) - n_before_bt
        logger.info(f"    [3] 离线回译: +{n_stage3} 条")

        # === 阶段4: 社交媒体噪声注入 ===
        n_before_noise = len(new_items)
        attempts = 0
        max_attempts = (n_needed - len(new_items)) * 3 + 1
        while len(new_items) < n_needed and attempts < max_attempts:
            source = random.choice(cls_data)
            protected = protector.get_protected_set(source)
            noise_texts = noise_aug.augment(source["content"], protected, n_aug=2)
            for aug_text in noise_texts:
                new_item = deepcopy(source)
                new_item["content"] = aug_text
                new_item["is_synthetic"] = True
                new_item["augmentation_method"] = "noise_injection"
                _try_add(new_item)
            attempts += 1

        n_stage4 = len(new_items) - n_before_noise
        logger.info(f"    [4] 噪声注入: +{n_stage4} 条")

        # === 阶段5: 句子级操作 ===
        n_before_sent = len(new_items)
        for item in cls_data:
            if len(new_items) >= n_needed:
                break
            protected = protector.get_protected_set(item)
            sent_texts = sentence_aug.augment(item["content"], protected, n_aug=2)
            for aug_text in sent_texts:
                new_item = deepcopy(item)
                new_item["content"] = aug_text
                new_item["is_synthetic"] = True
                new_item["augmentation_method"] = "sentence_level"
                _try_add(new_item)

        n_stage5 = len(new_items) - n_before_sent
        logger.info(f"    [5] 句子级操作: +{n_stage5} 条")

        # === 阶段6: 随机过采样补齐 ===
        if len(new_items) < n_needed:
            shortfall = n_needed - len(new_items)
            logger.info(f"    [6] 随机过采样补齐: +{shortfall} 条")
            for _ in range(shortfall):
                source = random.choice(cls_data)
                new_item = deepcopy(source)
                new_item["is_synthetic"] = True
                new_item["augmentation_method"] = "random_oversample"
                new_items.append(new_item)

        augmented_data.extend(new_items[:n_needed])
        logger.info(f"    Level {cls} 完成: {n_current} + "
                     f"{min(len(new_items), n_needed)} = "
                     f"{n_current + min(len(new_items), n_needed)} (目标 {cls_target})")

    # 最终统计
    final_counter = Counter(item["risk_level"] for item in augmented_data)
    logger.info(f"[管线] 最终分布: {dict(sorted(final_counter.items()))}")
    logger.info(f"[管线] 总样本: {len(augmented_data)} "
                f"(原始={len(data)}, 合成={len(augmented_data) - len(data)})")

    method_counter = Counter(
        item.get("augmentation_method", "original")
        for item in augmented_data
    )
    logger.info(f"[管线] 增强方法贡献: {dict(method_counter.most_common())}")

    return augmented_data


# ============================================================================
# 10. CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="无同义词表的暗语感知数据增强工具 (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法：

  基本用法（推荐）:
    python data_augmentation_v3.py \\
        --input annotated_risk_dataset.json \\
        --output augmented_dataset.json \\
        --evidence evidence/slang_emoji_dict.json

  启用回译增强（需要先安装 argostranslate）:
    pip install argostranslate
    python data_augmentation_v3.py \\
        --input annotated_risk_dataset.json \\
        --output augmented_dataset.json \\
        --evidence evidence/slang_emoji_dict.json \\
        --backtranslation

  然后用增强后的数据训练模型:
    python model_validation.py \\
        --input augmented_dataset.json \\
        --models tfidf_svm,tfidf_rf,llm_zeroshot

输入：标注后的数据集 JSON（每条记录含 content, risk_level, evidence_used 等字段）
输出：增强后的数据集 JSON（格式完全一样，少数类样本数量被补齐了）
        """,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="输入JSON文件路径（标注后的数据集）")
    parser.add_argument("--output", "-o", required=True,
                        help="输出JSON文件路径（增强后的数据集）")
    parser.add_argument("--evidence", "-e", default="evidence/slang_emoji_dict.json",
                        help="暗语证据库路径 (默认: evidence/slang_emoji_dict.json)")
    parser.add_argument("--target-count", type=int, default=None,
                        help="目标：让每个等级至少有多少条样本 (仅equal策略使用，其他策略自动计算)")
    parser.add_argument("--balance-strategy", type=str, default="sqrt",
                        choices=["equal", "sqrt", "log", "effective_number"],
                        help="分层平衡策略 (默认: sqrt)。"
                             "equal=统一到最大类; sqrt=平方根平衡(推荐); "
                             "log=对数平衡(温和); effective_number=有效样本数(Cui2019)")
    parser.add_argument("--balance-beta", type=float, default=0.99,
                        help="有效样本数策略的β参数 (默认: 0.99，仅effective_number使用)")
    parser.add_argument("--backtranslation", action="store_true",
                        help="启用离线回译增强（需要安装 argostranslate）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1. 加载数据 ----
    logger.info(f"[加载] 读取: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    data = [item for item in data if item.get("risk_level", -1) >= 0]
    logger.info(f"[加载] 有效样本: {len(data)}")

    counter = Counter(item["risk_level"] for item in data)
    logger.info(f"[加载] 原始分布: {dict(sorted(counter.items()))}")

    n_with_evidence = sum(1 for item in data
                          if item.get("evidence_used") and len(item["evidence_used"]) > 0)
    n_with_literal = sum(1 for item in data
                         if item.get("literal_meaning", "").strip())
    logger.info(f"[加载] 有evidence_used: {n_with_evidence}/{len(data)} "
                f"({n_with_evidence/len(data)*100:.1f}%)")
    logger.info(f"[加载] 有literal_meaning: {n_with_literal}/{len(data)} "
                f"({n_with_literal/len(data)*100:.1f}%)")

    # ---- 2. 执行增强 ----
    augmented = augmentation_pipeline(
        data,
        evidence_path=args.evidence,
        target_count=args.target_count,
        seed=args.seed,
        enable_backtranslation=args.backtranslation,
        balance_strategy=args.balance_strategy,
        balance_beta=args.balance_beta,
    )

    # ---- 3. 保存结果 ----
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    logger.info(f"[输出] 已保存到 {args.output}")
    logger.info(f"[提示] 下一步可直接用增强后的数据训练模型:")
    logger.info(f"  python model_validation.py --input {args.output} "
                f"--models tfidf_svm,tfidf_rf,llm_zeroshot")
    logger.info("[完成] 数据增强流程结束")


if __name__ == "__main__":
    main()