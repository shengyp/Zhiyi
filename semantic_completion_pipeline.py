import os
import json
import pandas as pd
import re
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pypinyin import lazy_pinyin, Style

try:
    import emoji as emoji_lib
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False
    print("[警告] emoji库未安装，emoji含义提取将被跳过。安装: pip install emoji")


# ===== 拼音工具函数 =====

def get_pinyin_key(text: str) -> str:
    pinyins = lazy_pinyin(text, style=Style.NORMAL, errors='default')
    return "".join(pinyins).lower().strip()


def find_pinyin_group(surface: str, evidence_data: list) -> dict:
    key = get_pinyin_key(surface)
    if not key:
        return None
    for item in evidence_data:
        if item.get("pinyin_key", "") == key:
            return item
    return None


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ===== Emoji 工具函数 =====

def extract_emoji_info(text):
    """从文本中提取emoji及其中文含义描述。"""
    if not HAS_EMOJI_LIB or not text:
        return []
    results = []
    seen = set()
    for item in emoji_lib.emoji_list(text):
        em = item["emoji"]
        if em in seen:
            continue
        seen.add(em)
        # 优先中文描述，回退英文
        try:
            desc = emoji_lib.demojize(em, language='zh').strip(':').replace('_', ' ')
        except (TypeError, KeyError):
            desc = emoji_lib.demojize(em).strip(':').replace('_', ' ')
        results.append({"emoji": em, "description": desc})
    return results


# ===== RAG 检索函数 =====

def rag_retrieve_for_slang(surface, guess_meaning, evidence_data):
    """
    RAG检索：为未命中证据库的隐语检索相关条目。
    检索维度：
      1. 拼音模糊匹配（字音相似，编辑距离阈值）
      2. 字形相似匹配（surface字符级编辑距离）
    返回: list[dict]，每项含 entry, match_type, similarity
    """
    retrieved = []
    py_key = get_pinyin_key(surface)

    # 1. 拼音模糊匹配（非精确同音，已由上游处理）
    if py_key:
        for entry in evidence_data:
            entry_py = entry.get("pinyin_key", "")
            if not entry_py or entry_py == py_key:
                continue  # 精确同音已处理，只找近似
            max_len = max(len(py_key), len(entry_py))
            if max_len == 0:
                continue
            dist = levenshtein_distance(py_key, entry_py)
            sim = 1.0 - dist / max_len
            if sim >= 0.6:
                retrieved.append({"entry": entry, "match_type": "pinyin_fuzzy", "similarity": round(sim, 3)})

    # 2. 字形相似匹配（surface字符级编辑距离，不限制等长）
    if len(surface) >= 1:
        for entry in evidence_data:
            for variant in entry.get("variants", [entry.get("surface", "")]):
                if variant == surface or not variant:
                    continue
                max_len = max(len(surface), len(variant))
                dist = levenshtein_distance(surface, variant)
                sim = 1.0 - dist / max_len
                if sim >= 0.5 and dist <= 3:
                    retrieved.append({"entry": entry, "match_type": "glyph", "similarity": round(sim, 3)})
                    break  # 同一entry只取一次

    # 去重（同一entry保留最高相似度），按相似度降序
    seen_ids = {}
    for r in retrieved:
        eid = id(r["entry"])
        if eid not in seen_ids or r["similarity"] > seen_ids[eid]["similarity"]:
            seen_ids[eid] = r
    unique = sorted(seen_ids.values(), key=lambda x: x["similarity"], reverse=True)
    return unique[:5]


# ===== 隐语证据库拼音归并（迁移工具） =====

def consolidate_evidence_db(evidence_path: str, dry_run: bool = False):
    with open(evidence_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    pinyin_groups = {}
    for item in old_data:
        surface = item.get("surface", "").strip()
        if not surface:
            continue
        key = get_pinyin_key(surface)
        if key not in pinyin_groups:
            pinyin_groups[key] = []
        pinyin_groups[key].append(item)

    new_data = []
    merge_count = 0
    for key, group in pinyin_groups.items():
        if len(group) == 1:
            entry = group[0].copy()
            entry["pinyin_key"] = key
            entry["variants"] = [entry["surface"]]
            new_data.append(entry)
        else:
            merge_count += len(group)
            primary = group[0]
            all_variants = list(dict.fromkeys(
                [g["surface"] for g in group]
            ))
            meanings = list(dict.fromkeys(
                [g.get("meaning", "") for g in group if g.get("meaning", "")]
            ))
            hints = list(dict.fromkeys(
                [g.get("risk_hint", "") for g in group if g.get("risk_hint", "")]
            ))
            new_data.append({
                "surface": primary["surface"],
                "pinyin_key": key,
                "variants": all_variants,
                "meaning": "; ".join(meanings) if meanings else "",
                "risk_hint": "; ".join(hints) if hints else "",
            })

    print(f"[拼音归并] 原始条数: {len(old_data)}, 归并后条数: {len(new_data)}, "
          f"其中 {merge_count} 条被合并为同音组")

    if not dry_run:
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print(f"[拼音归并] 已写入 {evidence_path}")
    else:
        print("[拼音归并] dry_run 模式，未写入文件")

    return new_data


# ===== 参数解析 =====

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="full",
                    choices=["full", "filter", "update_slang", "semantic", "consolidate"])
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--consolidate_dry_run", action="store_true")
args = parser.parse_args()


# ===== 初始化 =====

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

EVIDENCE_PATH = "evidence/slang_emoji_dict.json"

with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("output", run_id)
os.makedirs(RUN_DIR, exist_ok=True)

STAGE1_PATH = os.path.join(RUN_DIR, "stage1_filtered.json")
STAGE2_LOG_PATH = os.path.join(RUN_DIR, "stage2_translation_log.json")
STAGE3_PATH = os.path.join(RUN_DIR, "final_semantic_completion.json")
REVIEW_PATH = os.path.join(RUN_DIR, "uncertain_slang_review.json")

OUTPUT_FIELDS = {"content", "time", "like", "comment", "platform"}


# ===== 数据加载 =====

def find_file(root_dir, filename):
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} 未找到")


def load_weibo_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["weibo"])
    df["time"] = df["publish_time"]
    df["like"] = pd.to_numeric(df["up_num"], errors="coerce").fillna(0).astype(int)
    df["comment"] = pd.to_numeric(df["comment_num"], errors="coerce").fillna(0).astype(int)
    df = df[["content", "time", "like", "comment"]]
    df["platform"] = "weibo"
    return df


def load_xhs_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["title"] = df["title"].fillna("")
    df["desc"] = df["desc"].fillna("")
    def build_content(row):
        title = str(row["title"]).strip()
        desc = str(row["desc"]).strip()
        if title and desc:
            return title + "\n" + desc
        return title or desc or ""
    df["content"] = df.apply(build_content, axis=1)
    df["time"] = df["upload_time"]
    df["like"] = pd.to_numeric(df["liked_count"], errors="coerce").fillna(0).astype(int)
    df["comment"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)
    df = df[["content", "time", "like", "comment"]]
    df["platform"] = "xhs"
    df = df[df["content"].str.strip().astype(bool)]
    return df


def normalize_text(text):
    text = re.sub(r"#.*?#", "", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def deduplicate(df):
    df = df.drop_duplicates(subset=["content"])
    df["norm"] = df["content"].apply(normalize_text)
    df = df.drop_duplicates(subset=["norm"])
    df = df.drop(columns=["norm"])
    return df


# ===== 小红书预筛选 =====

def _ocr_coherence_score(text):
    if not text:
        return 0.0
    lines = text.split("\n")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text.strip()
    if not body:
        return 0.0
    score = 1.0
    repeat_pattern = re.findall(r'(.)\1{4,}', body)
    if len(repeat_pattern) >= 2:
        score -= 0.3
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', body))
    total_chars = len(re.sub(r'\s', '', body))
    if total_chars > 0:
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio < 0.2:
            score -= 0.4
    segments = re.split(r'[，。！？、；：\s,.\n]+', body)
    segments = [s for s in segments if s.strip()]
    if segments:
        avg_seg_len = sum(len(s) for s in segments) / len(segments)
        if avg_seg_len < 1.5 and len(segments) > 5:
            score -= 0.3
    garbage_chars = len(re.findall(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
                                    r'a-zA-Z0-9\s，。！？、；：""''（）【】'
                                    r'…—\-\.\,\!\?\:\;\'\"\(\)\[\]'
                                    r'\U0001f300-\U0001f9ff\U00002702-\U000027b0]', body))
    if total_chars > 0 and garbage_chars / total_chars > 0.15:
        score -= 0.3
    return max(score, 0.0)


def xhs_pre_filter(text):
    if not text or not text.strip():
        return False
    text_stripped = text.strip()
    lines = text_stripped.split("\n")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text_stripped
    if len(body) < 10:
        return False
    coherence = _ocr_coherence_score(text_stripped)
    if coherence < 0.5:
        return False
    text_lower = text_stripped.lower()
    bot_admin_signals = [
        "投稿方式", "投稿须知", "投稿格式", "如何投稿",
        "本账号", "本bot", "合集汇总", "目录", "置顶",
    ]
    admin_count = sum(1 for s in bot_admin_signals if s in text_lower)
    if admin_count >= 2:
        return False
    return True


# ===== LLM 筛选 + 隐语识别（合一调用） =====

def stage1_and_slang(text):
    """
    使用 LLM 同时完成内容筛选和隐语识别。
    筛选标准严格要求第一人称真实负面情绪，隐语识别仅针对非常规隐晦表达。
    """
    prompt = f"""任务1：判断以下社交媒体文本是否符合筛选标准。
必须**同时满足所有条件**才算通过：
1. 第一人称视角：文本主体是作者自己的经历、感受或想法，不是在讲述别人的故事（如"我朋友/同学/同事……"为主体的叙事），不是新闻报道或书评/影评
2. 真实负面情绪：表达了作者自己的真实的痛苦、绝望、悲伤、焦虑等负面情绪，或描述了自伤/自杀相关的意念和行为
3. 非建议或指导：不是基于自身经历或者医护人员视角给他人的生活建议和医疗指导

任务2：若符合筛选标准，请识别文本中与自我伤害、自杀、心理危机相关的隐语。
**隐语识别标准**：
- 只识别隐晦表达的非常规表达短语
- 如：谐音词（"紫砂"→自杀）、拼音缩写（"zs"→自杀）、emoji隐喻（🔪→自残）、字形变体（"亖"→死）
- **不要**将以下类型识别为隐语：
  · 日常生活常见词汇（如"崩溃"、"痛苦"、"难受"、"绝望"、"解脱"、"抑郁"等，这些是常规中文词汇，不是隐语）
  · 标准情绪表达词（如"焦虑"、"恐惧"、"悲伤"等）

本阶段只需识别隐语的原文形式，不需要翻译含义。
输出JSON：
{{
  "pass": true/false,
  "reject_reason": "若不通过，简述原因",
  "discovered_slang":[
    {{ "surface":"隐语原文" }}
  ]
}}
文本：{text}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是心理风险筛选与隐语识别专家。你的任务是严格按照标准筛选社交媒体文本，只保留作者本人表达真实负面情绪或自伤/自杀相关内容的帖子。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def llm_translate_with_rag(slang_list, text, evidence_data):
    """
    RAG增强的隐语翻译（Retrieve → Augment → Generate）。
    对未命中证据库的隐语，先从库中检索相似条目作为参考证据，
    再将检索结果注入LLM提示词中进行增强生成。
    """
    if not slang_list:
        return []

    # ---- Phase 1: Retrieve ----
    slang_context = []
    for s in slang_list:
        surface = s.get("surface", "")
        retrieved = rag_retrieve_for_slang(surface, "", evidence_data)
        emoji_info = extract_emoji_info(surface)
        slang_context.append({
            "slang": s, "retrieved": retrieved, "emoji_info": emoji_info
        })

    # ---- Phase 2: Augment (构建增强提示) ----
    slang_desc_parts = []
    for ctx in slang_context:
        s = ctx["slang"]
        lines = [f"隐语原文: \"{s['surface']}\""]

        # 注入emoji含义
        if ctx["emoji_info"]:
            emoji_str = "、".join([f"{e['emoji']}({e['description']})" for e in ctx["emoji_info"]])
            lines.append(f"  Emoji含义: {emoji_str}")

        # 注入检索到的相似证据
        if ctx["retrieved"]:
            lines.append("  证据库相似条目（供参考）:")
            for r in ctx["retrieved"][:5]:
                entry = r["entry"]
                variants = ", ".join(entry.get("variants", [entry["surface"]]))
                lines.append(f"    - {variants} → {entry.get('meaning', '未知')} "
                             f"[匹配: {r['match_type']}, 相似度: {r['similarity']:.2f}]")

        slang_desc_parts.append("\n".join(lines))

    augmented_evidence = "\n\n".join(slang_desc_parts)

    # ---- Phase 3: Generate ----
    prompt = f"""以下社交媒体文本中识别到了一些疑似隐语（非常规隐晦表达）。
系统已从隐语知识库中检索到了与这些隐语在字音、字形或emoji语义上相似的已知条目作为参考。
请结合原文上下文语境以及检索到的参考证据，为每个隐语提供：
1. 准确的语义翻译（该隐语在此语境中的实际含义）
2. 心理风险提示（该隐语暗示的心理风险，如表达什么样的情绪、情绪程度如何、是否暗示自伤/自杀倾向等）
3. 置信度（0-1之间，表示你对该翻译的确信程度）

原文：
{text}

识别到的隐语及检索证据：
{augmented_evidence}

输出JSON：
{{
  "translations": [
    {{
      "surface": "隐语原文",
      "guess_meaning": "该隐语的实际含义",
      "risk_hint_guess": "该隐语暗示的心理风险（一句话）",
      "confidence": 0.0-1.0
    }}
  ]
}}

注意：
- 如果某个检测结果明显不是隐语（是正常用词），请在 guess_meaning 中标注"非隐语"
- 充分利用检索到的相似条目作为翻译参考，但不要照搬，要根据上下文判断"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是心理风险隐语翻译专家，善于理解社交媒体中的隐晦表达。系统为你提供了从隐语知识库中检索到的相似条目作为参考，请结合这些参考和上下文给出准确翻译。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        translations = result.get("translations", [])
        if not isinstance(translations, list):
            translations = []

        # confidence 由 LLM 在翻译时直接给出
        for t in translations:
            t["confidence"] = float(t.get("confidence", 0.5))
            t["source"] = "rag_translate"

        # 过滤"非隐语"
        translations = [t for t in translations if "非隐语" not in t.get("guess_meaning", "")]

        # 构建检索日志（记录每条隐语的检索证据）
        retrieval_log = []
        for ctx in slang_context:
            log_entry = {
                "surface": ctx["slang"].get("surface", ""),
                "emoji_info": ctx["emoji_info"],
                "retrieved_evidence": []
            }
            for r in ctx["retrieved"]:
                entry = r["entry"]
                log_entry["retrieved_evidence"].append({
                    "surface": entry.get("surface", ""),
                    "variants": entry.get("variants", []),
                    "meaning": entry.get("meaning", ""),
                    "risk_hint": entry.get("risk_hint", ""),
                    "match_type": r["match_type"],
                    "similarity": r["similarity"]
                })
            retrieval_log.append(log_entry)

        return translations, retrieval_log
    except Exception as e:
        print(f"  [RAG翻译异常] {e}")
        return [], []


def retrieve_substring_evidence(text, evidence_data_ref=None):
    """
    精确子串匹配：遍历证据库所有条目的全部变体，
    按变体文本长度降序排列，逐一检查是否出现在目标文本中。
    匹配成功的条目进行条目级去重，确保同一条目不重复计入。
    """
    if evidence_data_ref is None:
        evidence_data_ref = evidence_data
    all_variant_pairs = []
    for item in evidence_data_ref:
        variants = item.get("variants", [item.get("surface", "")])
        for v in variants:
            if v:
                all_variant_pairs.append((v, item))
    # 按变体文本长度降序排列，优先匹配长变体
    all_variant_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    found_items = []
    found_entry_ids = set()
    for variant, entry in all_variant_pairs:
        if variant in text and id(entry) not in found_entry_ids:
            found_items.append(entry)
            found_entry_ids.add(id(entry))
    return found_items


def semantic_completion(text, translated_slang=None):
    """
    基于精确子串匹配从证据库中检索已知隐语，进行语义补全。
    遍历证据库全部变体，按长度降序匹配，条目级去重。
    """
    time.sleep(0.5)

    # 精确子串匹配：从证据库中检索文本中出现的已知隐语
    matched_entries = retrieve_substring_evidence(text)

    # 构建证据文本
    evidence_items = []
    for entry in matched_entries:
        surface = entry.get("surface", "")
        variants = entry.get("variants", [surface])
        meaning = entry.get("meaning", "")
        risk_hint = entry.get("risk_hint", "")
        if surface and meaning:
            variant_str = ", ".join(variants) if len(variants) > 1 else surface
            evidence_items.append(f"- 隐语: {variant_str}\n  释义: {meaning}\n  风险提示: {risk_hint or '无'}")

    evidence_text = "\n".join(evidence_items)

    prompt = f"""任务：基于提供的隐语证据对社交媒体文本进行分步语义解析。请严格按照以下两个任务依次执行。

待解析文本：
{text}

隐语证据（包含释义及风险背景）：
{evidence_text if evidence_text else "（未检测到隐语，请按常规理解分析）"}

任务1：语句改写
规则：
- 逐句检查原文，若某句包含隐语证据中列出的隐语（含其变体），则参考隐语对应的释义，重新组织语言表述出一个通顺、直白、无隐语的表达。
- 若文本中含有疑似隐语但证据库未匹配到，可根据上下文常识进行替换。
- 将改写后的完整文本作为 literal_meaning 输出。

任务2：结构化心理风险推断
基于任务1的改写结果与隐语证据对应风险提示，对以下三个维度分别做出判断，每个维度只能选择给定选项之一：

维度A - 自杀意念：
  "无"：文本未表现出任何自杀相关的想法
  "被动"：表达了想死、活着没意思等被动意愿，但无具体方法或计划
  "主动"：提及了具体的自杀方法、时间、地点或计划

维度B - 行为证据：
  "无"：未提及任何自伤或自杀相关行为
  "有准备行为"：提及了为自杀做准备（如囤积药物、写遗书、处理个人物品等），但尚未实施
  "有过尝试"：明确提及过去或当前已经实施的自伤或自杀未遂行为
  "迫在眉睫的行为"：提及了即将实施的自杀行为，或已经进入实施阶段

维度C - 情绪强度：
  "轻度"：表达了一定程度的负面情绪，但整体情绪尚可控
  "中度"：表达了较强的痛苦、绝望或无助，情绪明显失控
  "重度"：表达了极端的绝望、决绝感或情感麻木，呈现强烈危机信号

最后，基于以上三个维度的判断，撰写一段简要的综合推断（pragmatic_inference），概括作者当前的心理状态和风险程度。只需输出分析结论，无需复述证据内容。

输出JSON：
{{
  "literal_meaning": "任务1的改写结果（完整文本）",
  "suicidal_ideation": "无/被动/主动",
  "behavioral_evidence": "无/有准备行为/有过尝试/迫在眉睫的行为",
  "emotional_intensity": "轻度/中度/重度",
  "pragmatic_inference": "步骤2的综合推断文本",
  "evidence_used": ["实际使用的隐语1", "实际使用的隐语2"]
}}

注意：
1. evidence_used 只需包含隐语证据列表中确实被你在步骤1中采纳替换的隐语surface字段。
2. 三个维度的取值必须严格从给定选项中选择，不得自创选项。
3. 严禁脑补证据列表中不存在的隐语解释。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是资深心理健康语义分析专家，善于通过隐语识别文本背后的危机信号。请严格按照用户指定的步骤和格式执行任务。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content) if raw_content else {}

        # 清洗 evidence_used
        raw_evidence = result.get("evidence_used", [])
        if isinstance(raw_evidence, list):
            cleaned = []
            for item in raw_evidence:
                clean_name = str(item).split(':')[0].split('：')[0].split('-')[0].strip()
                if clean_name:
                    cleaned.append(clean_name)
            result["evidence_used"] = cleaned
        return result
    except Exception as e:
        print(f"API解析失败: {e}")
        return {}


# ===== 隐语证据库更新 =====

def update_evidence(slang_list, text):
    """
    更新证据库。高置信度自动入库，中置信度进入人工审核。
    已有隐语或同音变体不再累加 risk_hint，仅做变体归并。
    """
    global evidence_data
    surface_map = {}
    pinyin_map = {}
    for e in evidence_data:
        surface_map[e["surface"]] = e
        for v in e.get("variants", []):
            surface_map[v] = e
        pk = e.get("pinyin_key", "")
        if pk:
            pinyin_map[pk] = e

    review_items = []
    updated = False
    added_count = 0
    reviewed_count = 0

    existing_review = []
    if os.path.exists(REVIEW_PATH):
        try:
            with open(REVIEW_PATH, "r", encoding="utf-8") as f:
                existing_review = json.load(f)
        except:
            existing_review = []
    review_surface_set = set()
    for r in existing_review:
        review_surface_set.add(r.get("surface", ""))
        for v in r.get("variants", []):
            review_surface_set.add(v)

    for item in slang_list:
        surface = item.get("surface", "").strip()
        if not surface:
            continue
        meaning = item.get("guess_meaning", "")
        risk_hint = item.get("risk_hint_guess", "")
        conf = float(item.get("confidence", 0))
        py_key = get_pinyin_key(surface)

        if conf >= 0.85:
            # 已有条目（surface或variant命中）→ 跳过，不再累加 risk_hint
            if surface in surface_map:
                continue

            # 同音归并：拼音命中已有条目 → 添加变体，合并 meaning 和 risk_hint
            if py_key and py_key in pinyin_map:
                existing_entry = pinyin_map[py_key]
                if surface not in existing_entry.get("variants", []):
                    existing_entry.setdefault("variants", [existing_entry["surface"]])
                    existing_entry["variants"].append(surface)
                    surface_map[surface] = existing_entry
                    updated = True
                    print(f"  [拼音归并] '{surface}' 合并入已有条目 '{existing_entry['surface']}' (拼音: {py_key})")
                # 合并新的 meaning 和 risk_hint（去重追加）
                if meaning and meaning not in existing_entry.get("meaning", ""):
                    existing_entry["meaning"] = (existing_entry.get("meaning", "") + "; " + meaning).strip("; ")
                    updated = True
                if risk_hint and risk_hint not in existing_entry.get("risk_hint", ""):
                    existing_entry["risk_hint"] = (existing_entry.get("risk_hint", "") + "; " + risk_hint).strip("; ")
                    updated = True
                continue

            # 全新条目 → 入库
            new_entry = {
                "surface": surface,
                "pinyin_key": py_key,
                "variants": [surface],
                "meaning": meaning,
                "risk_hint": risk_hint
            }
            evidence_data.append(new_entry)
            surface_map[surface] = new_entry
            if py_key:
                pinyin_map[py_key] = new_entry
            updated = True
            added_count += 1

        elif 0.5 <= conf < 0.85:
            if surface in surface_map:
                continue
            if py_key and py_key in pinyin_map:
                continue
            if surface in review_surface_set:
                continue
            review_items.append({
                "surface": surface, "pinyin_key": py_key,
                "guess_meaning": meaning, "risk_hint_guess": risk_hint,
                "confidence": conf, "example_text": text
            })
            review_surface_set.add(surface)
            reviewed_count += 1

    if updated:
        with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
            json.dump(evidence_data, f, ensure_ascii=False, indent=2)

    if review_items:
        existing_review.extend(review_items)
        with open(REVIEW_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_review, f, ensure_ascii=False, indent=2)

    return added_count, reviewed_count


# ===== 预处理 =====

def preprocess_content(text):
    text = text.replace("/", "")
    text = re.sub(r'【\d+】', '', text)
    return text


# ===== 阶段执行函数 =====

def run_filter():
    """阶段1：预处理 + LLM 筛选帖子 + LLM 隐语识别。"""
    print("\n" + "=" * 60)
    print("  阶段1 -- 筛选帖子 + LLM隐语识别")
    print("=" * 60)

    weibo_path = find_file("data", "weibo_1997.json")
    xhs_path = find_file("data", "xhs_543.json")

    weibo_df = load_weibo_json(weibo_path)
    xhs_df = load_xhs_json(xhs_path)
    df = pd.concat([weibo_df, xhs_df], ignore_index=True).dropna(subset=["content"])
    df = deduplicate(df)

    

    # 预处理：去除规避审查的分隔符 "/" 和投稿序号 "【数字】"
    df["content"] = df["content"].apply(preprocess_content)

    weibo_input_count = len(weibo_df)
    xhs_input_count = len(xhs_df)

    weibo_after = len(df[df['platform'] == 'weibo'])
    xhs_after = len(df[df['platform'] == 'xhs'])

    print(f"\n[数据] 微博原始 : {weibo_input_count}")
    print(f"[数据] 小红书原始: {xhs_input_count}")
    print(f"[数据] 去重后微博 : {weibo_after}  (减少 {weibo_input_count - weibo_after} 条)")
    print(f"[数据] 去重后小红书: {xhs_after}  (减少 {xhs_input_count - xhs_after} 条)")
    print(f"[数据] 去重后合计 : {weibo_after + xhs_after}")

    # print(f"\n[数据] 微博原始: {weibo_input_count}  |  小红书原始: {xhs_input_count}  |  合计: {weibo_input_count + xhs_input_count}")
    # print(f"[数据] 去重后: {len(df)}\n")

    all_rows = []
    filtered_rows = []
    pre_filtered_count = 0
    llm_rejected_count = 0
    total_slang_discovered = 0

    print("--- 正在通过LLM筛选帖子并识别隐语 ---")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["content"])
        platform = row.get("platform", "")

        if platform == "xhs" and not xhs_pre_filter(text):
            pre_filtered_count += 1
            row_dict = row.to_dict()
            row_dict["pass"] = False
            row_dict["reject_reason"] = "预筛选过滤(小红书OCR质量/内容不合格)"
            row_dict["discovered_slang"] = []
            all_rows.append(row_dict)
            continue

        try:
            result = stage1_and_slang(text)
            row_dict = row.to_dict()
            if result.get("pass"):
                discovered = result.get("discovered_slang", [])
                row_dict["pass"] = True
                row_dict["reject_reason"] = ""
                row_dict["discovered_slang"] = discovered
                total_slang_discovered += len(discovered)
                filtered_rows.append(row_dict)
            else:
                llm_rejected_count += 1
                row_dict["pass"] = False
                row_dict["reject_reason"] = result.get("reject_reason", "LLM筛选未通过")
                row_dict["discovered_slang"] = []
            all_rows.append(row_dict)
        except Exception as e:
            row_dict = row.to_dict()
            row_dict["pass"] = False
            row_dict["reject_reason"] = f"处理异常: {e}"
            row_dict["discovered_slang"] = []
            all_rows.append(row_dict)
            continue

    with open(STAGE1_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # 统计
    total_content_chars = sum(len(str(row.get("content", ""))) for row in filtered_rows)
    total_slang_chars = 0
    for row in filtered_rows:
        for slang_item in row.get("discovered_slang", []):
            total_slang_chars += len(slang_item.get("surface", ""))
    slang_char_ratio = (total_slang_chars / total_content_chars * 100) if total_content_chars > 0 else 0.0

    print("\n" + "-" * 40)
    print(f"[筛选统计]")
    print(f"  预筛选过滤       : {pre_filtered_count}")
    print(f"  LLM拒绝          : {llm_rejected_count}")
    print(f"  保留（通过）      : {len(filtered_rows)}")
    print(f"  总输出(含未通过)  : {len(all_rows)}")
    print(f"  发现隐语数        : {total_slang_discovered}")
    print(f"  隐语字符数/帖文字符数 : {total_slang_chars}/{total_content_chars} ({slang_char_ratio:.2f}%)")
    print(f"  输出路径          : {STAGE1_PATH}")
    print("-" * 40)

    return filtered_rows


def run_update_slang(input_path):
    """
    阶段2：隐语翻译与证据库更新。
    流程：对每条帖文的检测隐语，先查证据库（精确/同音），
    未命中的走 RAG 检索 + LLM 增强翻译，最后更新证据库。
    输出翻译日志文件，记录所有参考的证据和最终翻译结果。
    """
    print("\n" + "=" * 60)
    print("  阶段2 -- RAG隐语翻译 + 更新隐语证据库")
    print("=" * 60)

    if not input_path:
        raise ValueError("update_slang 模式需要通过 --input 指定筛选后的JSON文件路径")

    with open(input_path, "r", encoding="utf-8") as f:
        all_rows = json.load(f)

    rows = [r for r in all_rows if r.get("pass", True)]

    total_added = 0
    total_reviewed = 0
    rows_with_slang = 0
    total_slang_items = 0
    total_db_hit = 0
    total_rag_translated = 0

    # 翻译日志：记录每条帖文中隐语的翻译过程和参考证据
    translation_log = []

    # 构建查找索引
    def _build_lookup():
        s_map, p_map = {}, {}
        for e in evidence_data:
            s_map[e["surface"]] = e
            for v in e.get("variants", []):
                s_map[v] = e
            pk = e.get("pinyin_key", "")
            if pk:
                p_map[pk] = e
        return s_map, p_map

    print(f"\n[输入] 从 {input_path} 加载了 {len(all_rows)} 条数据（其中通过筛选: {len(rows)} 条）")
    print("--- 正在翻译隐语并更新证据库 ---")

    for row in tqdm(rows, total=len(rows)):
        detected = row.get("discovered_slang", [])
        if not detected:
            continue
        rows_with_slang += 1
        total_slang_items += len(detected)
        text = row.get("content", "")

        surface_map, pinyin_map = _build_lookup()

        known_slang = []      # 已在库中（含同音归并）
        unknown_slang = []    # 需要 RAG + LLM 翻译
        row_log = {"content": text[:200], "slang_translations": []}

        for item in detected:
            surface = item.get("surface", "").strip()
            if not surface:
                continue

            # 1. 精确匹配（surface 或 variant）
            if surface in surface_map:
                entry = surface_map[surface]
                item["meaning"] = entry.get("meaning", "")
                item["risk_hint"] = entry.get("risk_hint", "")
                item["source"] = "db_exact"
                known_slang.append(item)
                total_db_hit += 1
                row_log["slang_translations"].append({
                    "surface": surface,
                    "source": "db_exact",
                    "matched_entry": entry.get("surface", ""),
                    "meaning": entry.get("meaning", ""),
                    "risk_hint": entry.get("risk_hint", ""),
                    "variants": entry.get("variants", [])
                })
                continue

            # 2. 精确同音匹配（pinyin_key 完全一致）
            py_key = get_pinyin_key(surface)
            if py_key and py_key in pinyin_map:
                entry = pinyin_map[py_key]
                # 归并为变体
                if surface not in entry.get("variants", []):
                    entry.setdefault("variants", [entry["surface"]])
                    entry["variants"].append(surface)
                    surface_map[surface] = entry
                    print(f"  [拼音归并] '{surface}' → '{entry['surface']}'")
                item["meaning"] = entry.get("meaning", "")
                item["risk_hint"] = entry.get("risk_hint", "")
                item["source"] = "db_pinyin"
                known_slang.append(item)
                total_db_hit += 1
                row_log["slang_translations"].append({
                    "surface": surface,
                    "source": "db_pinyin",
                    "matched_entry": entry.get("surface", ""),
                    "meaning": entry.get("meaning", ""),
                    "risk_hint": entry.get("risk_hint", ""),
                    "variants": entry.get("variants", [])
                })
                continue

            # 3. 未命中 → 送入 RAG 翻译队列
            unknown_slang.append(item)

        # 对未知隐语批量 RAG 翻译（一次 LLM 调用）
        if unknown_slang:
            translated, retrieval_log = llm_translate_with_rag(unknown_slang, text, evidence_data)
            total_rag_translated += len(translated)

            # 更新证据库
            added, reviewed = update_evidence(translated, text)
            total_added += added
            total_reviewed += reviewed

            known_slang.extend(translated)

            # 记录RAG翻译日志
            # 构建 retrieval_log 的 surface -> log 映射
            retrieval_by_surface = {r["surface"]: r for r in retrieval_log}
            for t in translated:
                t_surface = t.get("surface", "")
                log_entry = {
                    "surface": t_surface,
                    "source": "rag_translate",
                    "guess_meaning": t.get("guess_meaning", ""),
                    "risk_hint_guess": t.get("risk_hint_guess", ""),
                    "confidence": t.get("confidence", 0),
                }
                # 附加该隐语的检索证据
                r_log = retrieval_by_surface.get(t_surface, {})
                log_entry["emoji_info"] = r_log.get("emoji_info", [])
                log_entry["retrieved_evidence"] = r_log.get("retrieved_evidence", [])
                row_log["slang_translations"].append(log_entry)

        if row_log["slang_translations"]:
            translation_log.append(row_log)

    # 保存同音归并后的证据库
    with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, ensure_ascii=False, indent=2)

    # 保存翻译日志
    with open(STAGE2_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(translation_log, f, ensure_ascii=False, indent=2)

    total_groups = len(evidence_data)
    total_variants = sum(len(e.get("variants", [e.get("surface", "")])) for e in evidence_data)

    print("\n" + "-" * 40)
    print(f"[隐语更新统计]")
    print(f"  含隐语的行数      : {rows_with_slang} / {len(rows)}")
    print(f"  LLM识别隐语总数  : {total_slang_items}")
    print(f"  证据库直接命中    : {total_db_hit}")
    print(f"  RAG翻译生成       : {total_rag_translated}")
    print(f"  新增入库隐语      : {total_added}")
    print(f"  送入人工审核      : {total_reviewed}")
    print(f"  当前证据库语义组数 : {total_groups}")
    print(f"  共计包含隐语数量  : {total_variants}")
    print(f"  隐语证据库路径    : {EVIDENCE_PATH}")
    print(f"  翻译日志路径      : {STAGE2_LOG_PATH}")
    print("-" * 40)

    return rows


def run_semantic(input_path):
    """
    阶段3：语句重写。
    使用精确子串匹配从证据库中检索文本中出现的已知隐语作为证据，
    不再依赖阶段1+2中的 discovered_slang。
    """
    print("\n" + "=" * 60)
    print("  阶段3 -- 语义补全")
    print("=" * 60)

    if not input_path:
        raise ValueError("semantic 模式需要通过 --input 指定筛选后的JSON文件路径")

    with open(input_path, "r", encoding="utf-8") as f:
        stage_data = json.load(f)

    if stage_data and "pass" in stage_data[0]:
        stage_data = [r for r in stage_data if r.get("pass", True)]

    print(f"\n[输入] 从 {input_path} 加载了 {len(stage_data)} 条数据")

    final_results = []
    print("--- 正在执行语义补全 ---")
    for row in tqdm(stage_data, total=len(stage_data)):
        text = str(row.get("content", ""))

        # 使用精确子串匹配，直接从证据库检索
        semantic_result = semantic_completion(text)

        row_dict = {k: v for k, v in row.items() if k in OUTPUT_FIELDS}
        row_dict.update({
            "literal_meaning": semantic_result.get("literal_meaning", ""),
            "suicidal_ideation": semantic_result.get("suicidal_ideation", ""),
            "behavioral_evidence": semantic_result.get("behavioral_evidence", ""),
            "emotional_intensity": semantic_result.get("emotional_intensity", ""),
            "pragmatic_inference": semantic_result.get("pragmatic_inference", ""),
            "evidence_used": semantic_result.get("evidence_used", [])
        })
        final_results.append(row_dict)

    with open(STAGE3_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 40)
    print(f"[语义补全统计]")
    print(f"  处理行数          : {len(final_results)}")
    print(f"  输出保存至        : {STAGE3_PATH}")
    print("-" * 40)

    return final_results


# ===== 主函数 =====

def main():
    print(f"\n[流水线] 模式 = {args.mode}  |  运行ID = {run_id}")

    if args.mode == "consolidate":
        consolidate_evidence_db(EVIDENCE_PATH, dry_run=args.consolidate_dry_run)
        return
    elif args.mode == "filter":
        run_filter()
    elif args.mode == "update_slang":
        run_update_slang(args.input)
    elif args.mode == "semantic":
        run_semantic(args.input)
    elif args.mode == "full":
        run_filter()
        run_update_slang(STAGE1_PATH)
        run_semantic(STAGE1_PATH)

    print(f"\n[流水线] 完成。所有输出保存在: {RUN_DIR}\n")

if __name__ == "__main__":
    main()