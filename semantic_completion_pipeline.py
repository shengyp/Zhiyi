import os
import json
import faiss
import numpy as np
import pandas as pd
import re
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pypinyin import lazy_pinyin, Style


# 拼音工具函数：用于同音暗语归并
def get_pinyin_key(text: str) -> str:
    """
    将中文文本转换为无声调拼音键，用于同音暗语归并
    """
    pinyins = lazy_pinyin(text, style=Style.NORMAL, errors='default')
    return "".join(pinyins).lower().strip()


def find_pinyin_group(surface: str, evidence_data: list) -> dict:
    """
    在证据库中查找与给定暗语拼音相同的已有条目，只匹配 pinyin_key 相同的条目
    """
    key = get_pinyin_key(surface)
    if not key:
        return None
    for item in evidence_data:
        if item.get("pinyin_key", "") == key:
            return item
    return None


def consolidate_evidence_db(evidence_path: str, dry_run: bool = False):
    """
    迁移函数：将现有的逐条暗语库合并为拼音分组格式
    同拼音的多条暗语合并为一条，variants记录所有变体
    """
    with open(evidence_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    # 按拼音键分组
    pinyin_groups = {}
    for item in old_data:
        surface = item.get("surface", "").strip()
        if not surface:
            continue
        key = get_pinyin_key(surface)
        if key not in pinyin_groups:
            pinyin_groups[key] = []
        pinyin_groups[key].append(item)

    # 构建新的证据库
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
            # 代表词取第一个
            primary = group[0]
            all_variants = list(dict.fromkeys(
                [g["surface"] for g in group]
            ))  # 保序去重
            # 合并 meaning 和 risk_hint（去重拼接）
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


# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="full",
                    choices=["full", "filter", "update_slang", "semantic", "consolidate"])
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--consolidate_dry_run", action="store_true",
                    help="配合 --mode consolidate 使用，仅打印归并统计，不写入文件")
args = parser.parse_args()


# 初始化
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")
# 全局索引变量
index = None
_index_to_entry = []  # FAISS索引下标 -> evidence_data条目下标的映射

EVIDENCE_PATH = "evidence/slang_emoji_dict.json"

with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
    evidence_data = json.load(f)

def rebuild_faiss_index():
    """
    根据最新的 evidence_data 实时构建内存索引。
    如果条目含 variants 字段，则对所有变体分别编码，
    索引映射回对应的条目下标，确保任意变体都能被语义检索命中。
    """
    global index
    if not evidence_data:
        return

    # 收集所有需要编码的文本及其对应的条目下标
    texts_to_encode = []
    text_to_entry_idx = []  # 每个编码文本对应 evidence_data 中的下标
    for idx, item in enumerate(evidence_data):
        variants = item.get("variants", [])
        if variants:
            for v in variants:
                if v:
                    texts_to_encode.append(v)
                    text_to_entry_idx.append(idx)
        else:
            surface = item.get("surface", "")
            if surface:
                texts_to_encode.append(surface)
                text_to_entry_idx.append(idx)

    if not texts_to_encode:
        return

    embeddings = embed_model.encode(texts_to_encode).astype('float32')
    dim = embeddings.shape[1]
    new_index = faiss.IndexFlatL2(dim)
    new_index.add(embeddings)

    index = new_index
    # 将映射关系存为全局变量，供 retrieve_mixed_evidence 使用
    global _index_to_entry
    _index_to_entry = text_to_entry_idx

# 初始化索引
rebuild_faiss_index()

# 时间戳目录
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("output", run_id)
os.makedirs(RUN_DIR, exist_ok=True)

STAGE1_PATH = os.path.join(RUN_DIR, "stage1_filtered.json")         # 含 discovered_slang，供阶段2读取
STAGE1_CLEAN_PATH = os.path.join(RUN_DIR, "stage1_clean.json")      # 仅基础字段，供阶段3读取
STAGE3_PATH = os.path.join(RUN_DIR, "final_semantic_completion.json")
REVIEW_PATH = os.path.join(RUN_DIR, "uncertain_slang_review.json")

# 阶段1输出中保留给阶段3的基础字段（不含暗语猜测，避免干扰语义补全）
CLEAN_FIELDS = {"content", "time", "like", "comment", "platform"}


# 数据加载相关函数 (保持不变)
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
    """
    加载小红书JSON数据（OCR爬取格式）。
    字段：title（标题）、desc（OCR识别的正文）、liked_count、comment_count、upload_time等。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # title + desc(OCR正文) 拼接为 content
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
    # 过滤掉内容为空的行（OCR未识别到文字的笔记）
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


# 小红书内容预筛选
def _ocr_coherence_score(text):
    """
    评估OCR文本的连贯性/可读性分数（0-1）
    """
    if not text:
        return 0.0

    # 去除标题行，只检查正文
    lines = text.split("\n")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text.strip()
    if not body:
        return 0.0

    score = 1.0

    # 1. 连续重复字符检测
    repeat_pattern = re.findall(r'(.)\1{4,}', body)
    if len(repeat_pattern) >= 2:
        score -= 0.3

    # 2. 中文占比检测：正常中文文本中文字符占比应较高
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', body))
    total_chars = len(re.sub(r'\s', '', body))
    if total_chars > 0:
        chinese_ratio = chinese_chars / total_chars
        # 如果中文占比极低（<20%），可能是乱码或纯符号
        if chinese_ratio < 0.2:
            score -= 0.4

    # 3. 平均"词长"异常检测：OCR语序混乱时常产生大量单字碎片
    # 按标点和空格分割，看碎片密度
    segments = re.split(r'[，。！？、；：\s,.\n]+', body)
    segments = [s for s in segments if s.strip()]
    if segments:
        avg_seg_len = sum(len(s) for s in segments) / len(segments)
        # 正常中文段落平均片段长度通常 > 3
        if avg_seg_len < 1.5 and len(segments) > 5:
            score -= 0.3

    # 4. 特殊字符/乱码比例
    garbage_chars = len(re.findall(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
                                    r'a-zA-Z0-9\s，。！？、；：""''（）【】'
                                    r'…—\-\.\,\!\?\:\;\'\"\(\)\[\]'
                                    r'\U0001f300-\U0001f9ff\U00002702-\U000027b0]', body))
    if total_chars > 0 and garbage_chars / total_chars > 0.15:
        score -= 0.3

    return max(score, 0.0)


def xhs_pre_filter(text):
    """
    基于规则的快速预筛选，主要针对OCR数据质量问题。
    返回True表示通过预筛选，False表示直接过滤掉。

    数据来自精神疾病bot账号，内容已经过人工策展，不需要严格的关键词过滤。
    重点过滤：正文为空、OCR识别质量差导致的语序混乱/乱码。
    """
    if not text or not text.strip():
        return False

    text_stripped = text.strip()

    # 1. 正文长度检查：去除标题后，正文太短则视为无效
    lines = text_stripped.split("\n")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text_stripped
    if len(body) < 10:
        return False

    # 2. OCR连贯性检查
    coherence = _ocr_coherence_score(text_stripped)
    if coherence < 0.5:
        return False

    # 3. bot管理性内容（投稿须知、目录等非投稿内容）
    text_lower = text_stripped.lower()
    bot_admin_signals = [
        "投稿方式", "投稿须知", "投稿格式", "如何投稿",
        "本账号", "本bot", "合集汇总", "目录", "置顶",
    ]
    admin_count = sum(1 for s in bot_admin_signals if s in text_lower)
    if admin_count >= 2:
        return False

    return True


# 筛选内容+暗语识别（增强提示词）
def stage1_and_slang(text):
    # [修改] 更详细、更严格的筛选提示词
    prompt = f"""
任务1：判断以下社交媒体文本是否符合筛选标准。
必须**同时满足所有条件**才算通过：
1. 第一人称视角：文本主体是作者自己的经历、感受或想法（不是转述他人的故事）
2. 真实负面情绪：表达了作者自己的真实痛苦、绝望、悲伤、焦虑等负面情绪，或描述了自伤/自杀相关的意念和行为
3. 非转述：不是在讲述别人的故事（如"我朋友/同学/同事……"为主体的叙事）、不是新闻报道或书评/影评
4. 非求助/第三方救助：不是帮别人发的求助帖、不是作为旁观者发的求助帖
5. 非正能量鸡汤/生活医疗建议：不是自媒体营销号的正能量文章、不是给他人的建议和指导

任务2：若符合筛选标准，请识别文本中与自我伤害、自杀、心理危机相关的暗语。
**暗语识别标准**：
- 只识别隐晦表达的非常规表达短语
- 如：谐音词（"紫砂"→自杀）、拼音缩写（"zs"→自杀）、emoji隐喻（🔪→自残）、字形变体（"亖"→死）
- **不要**将以下类型识别为暗语：
  · 日常生活常见词汇（如"崩溃"、"痛苦"、"难受"、"绝望"、"解脱"、"抑郁"等，这些是常规中文词汇，不是暗语）
  · 标准情绪表达词（如"焦虑"、"恐惧"、"悲伤"等）
输出JSON：
{{
  "pass": true/false,
  "reject_reason": "若不通过，简述原因",
  "discovered_slang":[
    {{ "surface":"", "guess_meaning":"暗语的实际含义，用易于理解的方式简要描述", "risk_hint_guess":"分析暗语暗示的心理风险，如可能暗示自伤/自杀倾向或负面情绪、态度等，若仅作无风险用途也需进行说明", "confidence":0-1 }}
  ]
}}
文本：{text}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是心理风险筛选与隐语识别专家。你的任务是严格按照标准筛选社交媒体文本，只保留作者本人表达真实负面情绪或自伤/自杀相关内容的帖子。对于不确定的内容，宁可误过滤也不要放进来。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# 暗语库更新逻辑 (支持拼音归并去重和索引重构触发)
def update_evidence(slang_list, text):
    """
    更新证据库。高置信度词条自动加入，中置信度进入人工审核。
    使用拼音归并：如果新词与已有条目同音，则作为变体合并进已有条目，
    而非新增一条独立记录。
    """
    global evidence_data
    # 构建两种查找结构：按surface精确查找 + 按pinyin_key查找
    surface_map = {}  # surface -> entry
    pinyin_map = {}   # pinyin_key -> entry
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

    #  预加载人工审核库，用于去重检查
    existing_review = []
    if os.path.exists(REVIEW_PATH):
        try:
            with open(REVIEW_PATH, "r", encoding="utf-8") as f:
                existing_review = json.load(f)
        except:
            existing_review = []
    # 审核库也要按变体展开做去重
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
            # 高置信度：检查是否已有同surface或同拼音的条目
            if surface in surface_map:
                # 已存在（精确匹配或作为某组的变体）：仅追加新的风险提示
                existing_entry = surface_map[surface]
                if risk_hint and risk_hint not in existing_entry.get("risk_hint", ""):
                    existing_entry["risk_hint"] = existing_entry.get("risk_hint", "") + f"; {risk_hint}"
                    updated = True
            elif py_key and py_key in pinyin_map:
                # 拼音匹配：作为新变体合并进已有组
                existing_entry = pinyin_map[py_key]
                if surface not in existing_entry.get("variants", []):
                    existing_entry.setdefault("variants", [existing_entry["surface"]])
                    existing_entry["variants"].append(surface)
                    surface_map[surface] = existing_entry  # 更新查找表
                    updated = True
                    print(f"  [拼音归并] '{surface}' 合并入已有条目 '{existing_entry['surface']}' (拼音: {py_key})")
                if risk_hint and risk_hint not in existing_entry.get("risk_hint", ""):
                    existing_entry["risk_hint"] = existing_entry.get("risk_hint", "") + f"; {risk_hint}"
                    updated = True
            else:
                # 全新词条
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
            # 中等置信度：检查正式库和审核库（含变体和拼音去重）
            if surface in surface_map:
                continue
            if py_key and py_key in pinyin_map:
                # 拼音已在正式库中，跳过
                continue
            if surface in review_surface_set:
                continue
            review_items.append({
                "surface": surface,
                "pinyin_key": py_key,
                "guess_meaning": meaning,
                "risk_hint_guess": risk_hint,
                "confidence": conf,
                "example_text": text
            })
            review_surface_set.add(surface)
            reviewed_count += 1

    if updated:
        with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
            json.dump(evidence_data, f, ensure_ascii=False, indent=2)
        rebuild_faiss_index()  # 关键：更新后立即刷新内存索引

    if review_items:
        existing_review.extend(review_items)
        with open(REVIEW_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_review, f, ensure_ascii=False, indent=2)

    return added_count, reviewed_count


def retrieve_mixed_evidence(text, top_k=5, threshold=1.2):
    """
    混合检索：精确匹配 + 语义向量匹配。
    支持拼音分组格式：匹配任意变体即可命中对应条目。
    1. 精确包含匹配：文本中直接出现的暗语变体必选
    2. 语义向量匹配：补充语义相近的潜在词
    """
    found_items = []
    found_entry_ids = set()  # 用 id() 防止同一条目被重复添加

    # 1. 字符串包含匹配 (Precision优先)
    # 对每个条目，检查其所有变体是否出现在文本中
    # 按变体长度降序排列，防止短词干扰长词匹配
    all_variant_pairs = []  # (variant_text, entry)
    for item in evidence_data:
        variants = item.get("variants", [item.get("surface", "")])
        for v in variants:
            if v:
                all_variant_pairs.append((v, item))
    all_variant_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for variant, entry in all_variant_pairs:
        if variant in text and id(entry) not in found_entry_ids:
            found_items.append(entry)
            found_entry_ids.add(id(entry))

    # 2. 向量语义匹配 (Recall补充)
    if index is not None:
        try:
            query_vec = embed_model.encode([text]).astype('float32')
            D, I = index.search(query_vec, top_k + 5)

            for dist, idx in zip(D[0], I[0]):
                if idx < len(_index_to_entry):
                    entry_idx = _index_to_entry[idx]
                    target = evidence_data[entry_idx]
                    if id(target) not in found_entry_ids and dist < threshold:
                        found_items.append(target)
                        found_entry_ids.add(id(target))
                if len(found_items) >= top_k:
                    break
        except Exception as e:
            print(f"向量检索异常: {e}")

    return found_items[:top_k]


def semantic_completion(text):
    """
    分解锚定式语义补全：将原本的单一开放式生成任务拆分为两个子步骤，
    降低每步的自由度以提高输出置信度，仅需单次LLM调用。

    步骤1（受限改写）：仅将文本中已确认的暗语替换为其释义，其余部分保持原样。
    步骤2（结构化心理推断）：基于改写结果，对三个离散维度做枚举式判断，
          并输出综合推断文本。
    """
    time.sleep(0.5)
    # 使用混合检索获取包含 surface, meaning, risk_hint 的对象
    evidence_candidates = retrieve_mixed_evidence(text)

    # 构建证据信息文本，包含变体列表和释义
    evidence_items = []
    for e in evidence_candidates:
        variants = e.get("variants", [e["surface"]])
        variants_str = ", ".join(variants) if len(variants) > 1 else variants[0]
        item_str = f"- 暗语: {variants_str}\n  释义: {e['meaning']}\n  风险提示: {e.get('risk_hint', '无')}"
        evidence_items.append(item_str)

    evidence_text = "\n".join(evidence_items)

    prompt = f"""任务：基于提供的【暗语证据】对社交媒体文本进行分步语义解析。请严格按照以下两个步骤依次执行。

待解析文本：
{text}

暗语证据（包含释义及风险背景）：
{evidence_text if evidence_text else "（暂无匹配证据，请按常规理解分析）"}

━━━ 步骤1：语句改写 ━━━
规则：
- 逐句检查原文，若某句包含暗语证据中列出的暗语（含其变体），则参考暗语对应的释义，重新组织语言表述出一个通顺、直白、无暗语的表达。
- 若文本中含有疑似暗语但证据库未匹配到，可根据上下文常识进行替换。
- 将改写后的完整文本作为 literal_meaning 输出。

━━━ 步骤2：结构化心理风险推断 ━━━
基于步骤1的改写结果与暗语证据对应风险提示，对以下三个维度分别做出判断，每个维度只能选择给定选项之一：

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

━━━ 输出JSON格式 ━━━
{{
  "literal_meaning": "步骤1的改写结果（完整文本）",
  "suicidal_ideation": "无/被动/主动",
  "behavioral_evidence": "无/有准备行为/有过尝试/迫在眉睫的行为",
  "emotional_intensity": "轻度/中度/重度",
  "pragmatic_inference": "步骤2的综合推断文本",
  "evidence_used": ["实际使用的暗语1", "实际使用的暗语2"]
}}

注意：
1. evidence_used 只能包含暗语证据列表中确实被你在步骤1中采纳替换的暗语surface字段，严禁包含解释性文字。
2. 三个维度的取值必须严格从给定选项中选择，不得自创选项。
3. 严禁脑补证据列表中不存在的暗语解释。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是资深心理健康语义分析专家，善于通过暗语识别文本背后的危机信号。请严格按照用户指定的步骤和格式执行任务。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content) if raw_content else {}

        # 清洗 evidence_used：去除模型可能附加的解释性后缀
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


# ---------------------------------------------------------------------------
# 各阶段执行函数
# ---------------------------------------------------------------------------

def run_filter():
    """阶段1：筛选帖子并发现暗语（不更新暗语证据库）。"""
    print("\n" + "=" * 60)
    print("  阶段1 -- 筛选帖子 + 发现暗语")
    print("=" * 60)

    weibo_path = find_file("data", "weibo_1997.json")
    xhs_path = find_file("data", "xhs_543.json")

    weibo_df = load_weibo_json(weibo_path)
    xhs_df = load_xhs_json(xhs_path)
    df = pd.concat([weibo_df, xhs_df], ignore_index=True).dropna(subset=["content"])
    df = deduplicate(df)

    weibo_input_count = len(weibo_df)
    xhs_input_count = len(xhs_df)

    print(f"\n[数据] 微博原始: {weibo_input_count}  |  小红书原始: {xhs_input_count}  |  合计: {weibo_input_count + xhs_input_count}")
    print(f"[数据] 去重后: {len(df)}\n")

    all_rows = []          # 所有行（含通过和未通过），用于输出 stage1_filtered.json
    filtered_rows = []     # 仅通过筛选的行，用于后续阶段
    pre_filtered_count = 0
    llm_rejected_count = 0
    total_slang_discovered = 0

    print("--- 正在通过LLM筛选帖子 ---")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["content"])
        platform = row.get("platform", "")

        # 小红书内容先过规则预筛选
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
        except:
            row_dict = row.to_dict()
            row_dict["pass"] = False
            row_dict["reject_reason"] = "处理异常"
            row_dict["discovered_slang"] = []
            all_rows.append(row_dict)
            continue

    # 输出包含所有行（通过+未通过）的完整文件
    with open(STAGE1_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # 输出一份干净版本（仅通过筛选的行、仅基础字段），供阶段3语义补全使用，避免暗语猜测干扰LLM
    clean_rows = [
        {k: v for k, v in row.items() if k in CLEAN_FIELDS}
        for row in filtered_rows
    ]
    with open(STAGE1_CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_rows, f, ensure_ascii=False, indent=2)

    # 计算暗语字符占比：所有发现的暗语surface总字符数 / 通过筛选帖文总字符数
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
    print(f"  发现暗语数        : {total_slang_discovered}")
    print(f"  暗语字符数/帖文字符数 : {total_slang_chars}/{total_content_chars} ({slang_char_ratio:.2f}%)")
    print(f"  完整输出(含暗语)  : {STAGE1_PATH}")
    print(f"  干净输出(供阶段3) : {STAGE1_CLEAN_PATH}")
    print("-" * 40)

    return filtered_rows


def run_update_slang(input_path):
    """加载筛选后的数据，将发现的暗语更新到证据库。"""
    print("\n" + "=" * 60)
    print("  阶段2 -- 更新暗语证据库")
    print("=" * 60)

    if not input_path:
        raise ValueError("update_slang 模式需要通过 --input 指定筛选后的JSON文件路径")

    with open(input_path, "r", encoding="utf-8") as f:
        all_rows = json.load(f)

    # 仅处理通过筛选的行（兼容旧格式：无pass字段时视为通过）
    rows = [r for r in all_rows if r.get("pass", True)]

    total_added = 0
    total_reviewed = 0
    rows_with_slang = 0

    # 统计各置信度范围的暗语数量
    conf_high_count = 0    # >= 0.85
    conf_medium_count = 0  # 0.5 <= conf < 0.85
    conf_low_count = 0     # < 0.5
    total_slang_items = 0

    print(f"\n[输入] 从 {input_path} 加载了 {len(all_rows)} 条数据（其中通过筛选: {len(rows)} 条）")
    print("--- 正在更新暗语证据库 ---")

    for row in tqdm(rows, total=len(rows)):
        discovered = row.get("discovered_slang", [])
        if not discovered:
            continue
        rows_with_slang += 1

        # 统计各置信度范围
        for item in discovered:
            conf = float(item.get("confidence", 0))
            total_slang_items += 1
            if conf >= 0.85:
                conf_high_count += 1
            elif conf >= 0.5:
                conf_medium_count += 1
            else:
                conf_low_count += 1

        added, reviewed = update_evidence(discovered, row.get("content", ""))
        total_added += added
        total_reviewed += reviewed

    # 计算各置信度占比
    high_pct = (conf_high_count / total_slang_items * 100) if total_slang_items > 0 else 0.0
    medium_pct = (conf_medium_count / total_slang_items * 100) if total_slang_items > 0 else 0.0
    low_pct = (conf_low_count / total_slang_items * 100) if total_slang_items > 0 else 0.0

    total_groups = len(evidence_data)
    total_variants = sum(len(e.get("variants", [e.get("surface", "")])) for e in evidence_data)

    print("\n" + "-" * 40)
    print(f"[暗语更新统计]")
    print(f"  含暗语的行数      : {rows_with_slang} / {len(rows)}")
    print(f"  发现暗语总数      : {total_slang_items}")
    print(f"  置信度分布:")
    print(f"    高 (>=0.85)     : {conf_high_count} ({high_pct:.1f}%) -> 自动入库")
    print(f"    中 (0.5~0.85)   : {conf_medium_count} ({medium_pct:.1f}%) -> 人工审核")
    print(f"    低 (<0.5)       : {conf_low_count} ({low_pct:.1f}%) -> 丢弃")
    print(f"  新增入库暗语      : {total_added}")
    print(f"  送入人工审核      : {total_reviewed}")
    print(f"  当前证据库语义组数量    : {total_groups}")
    print(f"  共计包含暗语数量      : {total_variants}")
    print(f"  暗语证据库路径    : {EVIDENCE_PATH}")
    print("-" * 40)

    return rows


def run_semantic(input_path):
    """阶段3：对筛选后的数据执行RAG语义补全。"""
    print("\n" + "=" * 60)
    print("  阶段3 -- RAG语义补全")
    print("=" * 60)

    if not input_path:
        raise ValueError("semantic 模式需要通过 --input 指定筛选后的JSON文件路径")

    with open(input_path, "r", encoding="utf-8") as f:
        stage_data = json.load(f)

    stage_df = pd.DataFrame(stage_data)
    print(f"\n[输入] 从 {input_path} 加载了 {len(stage_df)} 条数据")

    final_results = []
    print("--- 正在执行RAG语义补全 ---")
    for _, row in tqdm(stage_df.iterrows(), total=len(stage_df)):
        text = str(row["content"])
        semantic_result = semantic_completion(text)

        row_dict = row.to_dict()
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


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    print(f"\n[流水线] 模式 = {args.mode}  |  运行ID = {run_id}")

    if args.mode == "consolidate":
        # ------- 拼音归并模式：将现有暗语库按同音合并 -------
        consolidate_evidence_db(EVIDENCE_PATH, dry_run=args.consolidate_dry_run)
        return

    elif args.mode == "filter":
        # ------- 仅筛选 -------
        run_filter()

    elif args.mode == "update_slang":
        # ------- 仅更新暗语库 -------
        run_update_slang(args.input)

    elif args.mode == "semantic":
        # ------- 仅语义补全 -------
        run_semantic(args.input)

    elif args.mode == "full":
        # ------- 完整流水线: 筛选 -> 更新暗语 -> 语义补全 -------
        filtered_rows = run_filter()

        # 阶段2用含暗语的完整版本
        run_update_slang(STAGE1_PATH)

        # 暗语更新后重建索引，让语义阶段使用最新证据
        rebuild_faiss_index()

        # 阶段3用干净版本（仅基础字段），避免暗语猜测干扰语义补全
        run_semantic(STAGE1_CLEAN_PATH)

    print(f"\n[流水线] 完成。所有输出保存在: {RUN_DIR}\n")

if __name__ == "__main__":
    main()