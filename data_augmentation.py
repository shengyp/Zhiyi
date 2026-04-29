"""
数据增强策略（v3-simplified）：
  1. 暗语互换：证据库中同义暗语互相替换（紫砂↔zs）
  2. 语义字段复用：literal_meaning作为独立样本、pragmatic拼接
  3. 离线回译：对literal_meaning做 中→英→中 回译产生自然变体
  4. 随机过采样补齐

类别平衡：平方根平衡（Mahajan et al., ECCV 2018）
  target_i = ceil(sqrt(n_i * n_max))

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
from typing import List, Dict, Optional, Any, Set
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
# 1. 动态保护词构建器
# ============================================================================

class DynamicProtectedWords:
    """
    从暗语证据库和样本的evidence_used字段动态构建保护词集合。
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
        meaning = self.slang_to_meaning.get(slang, "")
        if not meaning:
            return []
        return [s for s in self.meaning_to_slangs.get(meaning, []) if s != slang]
# ============================================================================
# 2. 暗语替换增强器
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
# 3. 语义字段复用增强器
# ============================================================================

class SemanticFieldAugmenter:
    """
    利用RAG阶段已生成的语义补全字段进行数据增强。

    策略a: literal_meaning作为独立样本
    策略b: 原文 + pragmatic_inference追加
    """

    def augment(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
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

        # 策略b: 原文 + pragmatic_inference追加
        if pragmatic and len(pragmatic) >= 15:
            enriched = f"{content} {pragmatic}"
            new_item = deepcopy(item)
            new_item["content"] = enriched
            new_item["is_synthetic"] = True
            new_item["augmentation_method"] = "pragmatic_enriched"
            results.append(new_item)

        return results
# ============================================================================
# 4. 离线回译增强器（argostranslate zh→en→zh）
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
    如果环境没有argostranslate，该增强器会静默跳过。
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
        if not self.available or not text.strip():
            return None
        try:
            en_text = self._translate_zh_en.translate(text)
            zh_back = self._translate_en_zh.translate(en_text)
            if zh_back.strip() == text.strip():
                return None
            return zh_back.strip()
        except Exception:
            return None

    def augment(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            new_item["evidence_used"] = []
            results.append(new_item)

        return results
# ============================================================================
# 5. 平方根平衡：计算每类目标样本数
# ============================================================================

def compute_sqrt_targets(class_counts: Dict[int, int]) -> Dict[int, int]:
    """
    平方根平衡 (Mahajan et al., ECCV 2018)。
    在完全平衡和原始分布之间取几何平均：
      target_i = ceil(sqrt(n_i * n_max))
    少数类大幅增强，但不会增强到与多数类一样多，
    保留了一部分自然分布的先验信息。
    """
    n_max = max(class_counts.values())
    targets = {}
    for cls, n in class_counts.items():
        targets[cls] = max(n, math.ceil(math.sqrt(n * n_max)))
    return targets


# ============================================================================
# 6. 组合增强管线
# ============================================================================

def augmentation_pipeline(
    data: List[Dict[str, Any]],
    evidence_path: Optional[str] = None,
    seed: int = 42,
    enable_backtranslation: bool = False,
) -> List[Dict[str, Any]]:
    """
    暗语感知组合增强管线（平方根平衡）。

    执行顺序（优先级递减）:
    1. 暗语互换（证据库中同义暗语替换，如紫砂↔zs）
    2. 语义字段复用（literal_meaning独立样本、pragmatic拼接）
    3. 离线回译（对literal_meaning做中→英→中，可选）
    4. 随机过采样补齐
    """
    random.seed(seed)
    np.random.seed(seed)

    # 初始化组件
    protector = DynamicProtectedWords(evidence_path)
    slang_aug = SlangSubstitutionAugmenter(protector)
    semantic_aug = SemanticFieldAugmenter()
    bt_aug = BackTranslationAugmenter() if enable_backtranslation else None

    # 按类别分组
    class_groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in data:
        class_groups[item["risk_level"]].append(item)

    counter = {k: len(v) for k, v in sorted(class_groups.items())}
    class_targets = compute_sqrt_targets(counter)

    logger.info(f"[管线] 原始分布: {counter}")
    logger.info(f"[管线] 平衡策略: sqrt (平方根平衡)")
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
        logger.info(f"  Level {cls}: {n_current} -> 目标 {cls_target}, "
                     f"需生成 {n_needed} 条")

        seen_texts = set()
        new_items = []

        def _try_add(item: Dict[str, Any]) -> bool:
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
            for r in semantic_aug.augment(item):
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

        # === 阶段4: 随机过采样补齐 ===
        if len(new_items) < n_needed:
            shortfall = n_needed - len(new_items)
            logger.info(f"    [4] 随机过采样补齐: +{shortfall} 条")
            for _ in range(shortfall):
                source = random.choice(cls_data)
                new_item = deepcopy(source)
                new_item["is_synthetic"] = True
                new_item["augmentation_method"] = "random_oversample"
                new_items.append(new_item)

        augmented_data.extend(new_items[:n_needed])
        logger.info(f"    Level {cls} 完成: {n_current} + "
                     f"{min(len(new_items), n_needed)} = "
                     f"{n_current + min(len(new_items), n_needed)} "
                     f"(目标 {cls_target})")

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
# 7. CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="暗语感知数据增强工具 (v3-simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="输入JSON文件路径（标注后的数据集）")
    parser.add_argument("--output", "-o", required=True,
                        help="输出JSON文件路径（增强后的数据集）")
    parser.add_argument("--evidence", "-e", default="evidence/slang_emoji_dict.json",
                        help="暗语证据库路径 (默认: evidence/slang_emoji_dict.json)")
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
                          if item.get("evidence_used")
                          and len(item["evidence_used"]) > 0)
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
        seed=args.seed,
        enable_backtranslation=args.backtranslation,
    )

    # ---- 3. 保存结果 ----
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    logger.info(f"[输出] 已保存到 {args.output}")
    logger.info("[完成] 数据增强流程结束")


if __name__ == "__main__":
    main()
