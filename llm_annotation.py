import os
import re
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

IPTS_ARCHETYPES = {
    "PB": {
        "name": "感知负担感 (Perceived Burdensomeness)",
        "description": "个体认为自己是他人的负担，世界没有自己会更好",
        "prototypes": [
            "大家没有我会过得更好",
            "我是家人的累赘和负担",
            "我活着只会拖累身边的人",
            "我对这个世界毫无价值",
            "我的存在对任何人来说都是多余的",
            "如果我消失了没人会在意",
            "我只会给别人添麻烦",
        ]
    },
    "TB": {
        "name": "受挫归属感 (Thwarted Belongingness)",
        "description": "个体感到孤立、不被接纳、与社会关系断裂",
        "prototypes": [
            "没有人真正关心我",
            "我在哪里都格格不入",
            "我是完全孤独的一个人",
            "所有人都讨厌我远离我",
            "我永远无法融入任何群体",
            "这个世界上没有一个人爱我",
            "我和所有人之间都隔着一堵墙",
        ]
    },
    "AC_IS": {
        "name": "习得能力-意念与模拟 (AC: Ideation/Simulation)",
        "description": "对自杀方式的具象化思考、想象和模拟",
        "prototypes": [
            "我想过从高处跳下去会是什么感觉",
            "我在脑海里反复演练过怎么结束一切",
            "我想过如果服药过量会发生什么",
            "我计划好了要用什么方式离开",
            "我已经想好了具体的时间和地点",
            "我在想用什么方式走最不痛苦",
        ]
    },
    "AC_EoE": {
        "name": "习得能力-忍耐经历 (AC: Experiences of Endurance)",
        "description": "经历过大量痛苦后产生的麻木和耐受",
        "prototypes": [
            "经历了太多痛苦我已经什么都不怕了",
            "我已经习惯了承受这些苦难",
            "再多的打击对我来说都无所谓了",
            "我已经被生活折磨到没有感觉了",
            "痛苦已经是我的日常了",
        ]
    },
    "AC_DtH": {
        "name": "习得能力-伤害脱敏 (AC: Desensitization to Harm)",
        "description": "对自伤行为的恐惧感降低",
        "prototypes": [
            "割伤自己的时候已经感觉不到害怕了",
            "看到血流出来反而觉得平静",
            "已经习惯了伤害自己这件事",
            "自伤对我来说已经变得很日常了",
            "每次难受就会不自觉地伤害自己",
        ]
    },
    "AC_HTPP": {
        "name": "习得能力-高疼痛耐受 (AC: High Tolerance for Physical Pain)",
        "description": "对身体疼痛的耐受程度提高",
        "prototypes": [
            "只有疼痛让我觉得自己还活着",
            "我已经感觉不到痛了",
            "身体上的疼痛对我来说已经不算什么",
            "割得越来越深才能有感觉",
            "痛觉好像已经消失了一样",
        ]
    },
    "AC_ERB": {
        "name": "习得能力-冒险行为 (AC: Engagement in Risky Behaviors)",
        "description": "反复接触自伤或危险行为，恐惧感降低",
        "prototypes": [
            "一崩溃就想割自己",
            "吃药过量好几次了已经不害怕了",
            "自杀未遂好几次了对死亡已经不恐惧了",
            "鬼门关走了好几遭已经无所谓了",
            "反复尝试过很多次了",
        ]
    },
    "AC_FSHM": {
        "name": "习得能力-熟悉自伤方法 (AC: Familiarity with Self-Harm Methods)",
        "description": "对自伤/自杀方法的了解和熟悉",
        "prototypes": [
            "我研究过各种结束生命的方式",
            "我知道哪种方法最有效",
            "我了解过不同药物过量的后果",
            "我清楚地知道怎样做才能成功",
            "我已经掌握了具体的方法和步骤",
        ]
    },
}

CSSRS_LEVEL_ARCHETYPES = {
    0: {
        "label": "负面情绪（无自杀风险）",
        "description": "表达日常负面情绪（压力、悲伤、焦虑、孤独等），未涉及死亡愿望、自伤意念或自伤行为",
        "prototypes": [
            "今天心情不太好，工作压力好大",
            "和朋友吵架了好烦躁，不想理任何人",
            "失眠好几天了整个人很疲惫什么都提不起劲",
            "最近状态很差什么都不想做只想躺着",
            "生活好无聊每天重复一样的事情",
            "考试挂科了心情很差觉得自己好没用",
            "被骂了一顿好郁闷好委屈",
            "感觉自己好焦虑做什么都做不好",
            "每天加班到深夜身体快撑不住了",
            "和对象分手了好难过一直在哭",
        ]
    },
    1: {
        "label": "被动死亡愿望",
        "description": "表达希望死亡、不想存在等被动愿望，但无主动结束生命的想法；行为上可能出现自我封闭、放弃治疗等消极应对，但无自伤行为",
        "prototypes": [
            "活着好累如果能消失就好了",
            "好想从这个世界上消失不被任何人记得",
            "每天醒来都觉得又要熬一天不想醒来了",
            "如果有个按钮能让我不存在我会毫不犹豫按下去",
            "活着没有任何意义只是在消耗时间而已",
            "感觉自己是多余的存在不存在都不会有人在意",
            "不想出门不想见人不想回复任何消息只想缩在角落",
            "药已经不想吃了治不治都无所谓了反正也没意义",
            "好羡慕那些不用醒来的人",
            "停了所有的药也不想去复诊了活不活随便吧",
        ]
    },
    2: {
        "label": "非特异性主动自杀意念",
        "description": "出现了主动的自杀表述（如'想死''不想活了'），但未涉及具体方法；行为上可能开始出现非自杀性自伤（NSSI），但无自杀性质的准备行为",
        "prototypes": [
            "不想活了真的好想死",
            "脑海里反复出现想死的念头控制不住",
            "有时候真的很想一了百了什么都不管了",
            "越来越频繁地想到死亡觉得死了就不用受苦了",
            "一直在想要不要结束这一切但又害怕",
            "不知道为什么脑子里总是想着去死甩不掉这个念头",
            "情绪崩溃的时候开始划自己几下才能冷静下来",
            "开始尝试割手了，不深但割完会好受一点",
            "有那么几个瞬间真的很想死掉然后情绪过了又犹豫",
            "开始用掐自己或者咬自己的方式来缓解焦虑",
        ]
    },
    3: {
        "label": "具有方法的自杀意念",
        "description": "开始考虑具体自杀手段（如药物、跳楼等），但尚未形成包含时间地点的完整计划；行为上可能出现试探性准备行为或频繁依赖性自伤",
        "prototypes": [
            "想过吃药走但不确定吃什么药有用",
            "路过高楼的时候会忍不住往上看想着从那里跳下去",
            "在网上搜过哪种方式自杀最不痛苦",
            "看到刀具会不自觉地想到可以用它结束自己",
            "想过去某个特定的地方了结但还没下定决心",
            "已经开始依赖自残来度过每一天不割就受不了",
            "手臂上的伤口越来越多越来越深了根本停不下来",
            "有好几种方式在脑子里反复比较但没想好选哪个",
            "去看过那座桥了想过从那里跳下去但只是看了一眼",
            "自残已经变成每天的习惯了不划就觉得无法入睡",
        ]
    },
    4: {
        "label": "具有计划的自杀意念",
        "description": "已形成包含时间、地点、方式的较明确实施方案；行为上存在明确准备行为或已发生被中止/被阻止的自杀未遂",
        "prototypes": [
            "已经想好了时间地点和方式都安排好了",
            "打算下周末用吃药的方式走已经确定了具体哪种药",
            "开始写遗书了把想对每个人说的话都写下来",
            "把手机里该删的都删了银行卡密码发给了家人",
            "已经偷偷攒了几十片安眠药藏在抽屉里",
            "给自己定了一个日期之前把所有事情处理完",
            "上次站在窗户边差一点就跳了被室友拉回来了",
            "吃了一把药但量不够被发现送去医院催吐了",
            "这几天在做最后的安排整理东西处理账号",
            "试过一次但没成功中途自己退缩了",
        ]
    },
    5: {
        "label": "具有明确意图的自杀意念或自杀未遂",
        "description": "具备完整计划且有强烈实施动机和决心；已发生实际自杀未遂或正处于即刻实施的危机状态",
        "prototypes": [
            "这是我发的最后一条消息和所有人告别",
            "药已经吃下去了感觉意识在慢慢模糊",
            "已经站在天台边上了这次不会犹豫了",
            "上次od进了icu抢救了三天醒来之后很后悔没死成",
            "这次选了最致命的方式不会再有意外了",
            "吃了一整瓶药被送去洗胃在icu住了一周",
            "再见了所有人遗书已经写好了放在桌上",
            "这次一定要成功上次失败是因为方式选错了",
            "正在从高处往下看马上就要结束这一切了",
            "绳子已经系好了倒计时开始",
        ]
    },
}

NLI_LEVEL_HYPOTHESES = {
    0: [
        "此人表达了日常生活中的负面情绪如压力、悲伤或焦虑，但没有任何关于死亡、消失或伤害自己的想法",
        "此人的情绪困扰停留在生活事件层面，完全没有涉及不想活、想死或自伤的表述",
    ],
    1: [
        "此人表达了希望自己不存在或对活着感到无意义的被动愿望，但没有主动说出'想死'或'想自杀'的话，也没有提到任何伤害自己的行为",
        "此人流露出被动的死亡愿望或厌世情绪，同时可能出现封闭自我、放弃治疗等消极行为，但没有主动的自杀念头，也没有自伤行为",
    ],
    2: [
        "此人主动表达了想死或想结束生命的念头，但没有提到具体的自杀方式或计划；可能提到了通过划伤自己等非自杀性自伤行为来应对情绪",
        "此人出现了反复的主动自杀想法但尚未考虑具体手段，行为上可能开始出现轻度自伤但没有为自杀做任何准备",
    ],
    3: [
        "此人开始考虑具体的自杀方式如吃药或跳楼，但尚未形成包含时间和地点的完整计划；可能出现了搜索自杀方法等试探行为或频繁的、较为严重的非自杀性自伤行为",
        "此人的自杀意念已经涉及具体手段的考量，行为上可能出现了信息搜集等准备迹象，或明显的对于非自杀性自伤行为的依赖，但还没有确定实施方案",
    ],
    4: [
        "此人已经形成了包含具体时间、地点和方式的自杀计划，并开始实施准备行为如囤积药物、撰写遗书或处理个人事务；或已经发生过在进行致死性行为前被中止的自杀未遂",
        "此人有明确的自杀实施方案并已着手准备，或曾试图自杀但被阻止或中途自己放弃，致死性行为尚未发生完整的自杀未遂行为",
    ],
    5: [
        "此人已经实际实施过自杀行为如服药过量后被送医抢救，或当前正处于即将实施自杀的即刻危机状态",
        "此人已发生自杀未遂被送医救治，或正在发出最后告别信号并表现出立即执行的强烈决心和具体行动",
    ],
}


class ArchetypeScorer:
    """
    使用literal_meaning字段计算余弦相似度（仅IPTS维度）
    """

    def __init__(self, model_name="shibing624/text2vec-base-chinese"):
        from sentence_transformers import SentenceTransformer
        print(f"[ArchetypeScorer] 加载句向量模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._encode_prototypes()

    def _encode_prototypes(self):
        """预编码 IPTS 原型语句向量"""
        self.ipts_keys = list(IPTS_ARCHETYPES.keys())
        self.ipts_proto_embeddings = {}
        for key in self.ipts_keys:
            protos = IPTS_ARCHETYPES[key]["prototypes"]
            self.ipts_proto_embeddings[key] = self.model.encode(
                protos, normalize_embeddings=True, show_progress_bar=False
            )
        print(f"[ArchetypeScorer] IPTS原型编码完成: {len(self.ipts_keys)}维")

    def encode_text(self, text):
        """编码单条文本。"""
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]

    def score_ipts(self, text_embedding):
        """计算文本向量与8个IPTS维度原型的平均余弦相似度。"""
        scores = {}
        for key in self.ipts_keys:
            proto_embs = self.ipts_proto_embeddings[key]
            sims = np.dot(proto_embs, text_embedding)
            scores[key] = float(np.mean(sims))
        return scores


class NLIScorer:
    """
    使用多语言NLI模型判断pragmatic_inference是否蕴含各等级描述
    """

    def __init__(self, model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.torch = torch
        print(f"[NLIScorer] 加载NLI模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.label2id = self.model.config.label2id
        self.entail_idx = self.label2id.get("entailment", 0)
        print(f"[NLIScorer] 模型加载完成, 设备: {self.device}, "
              f"entailment索引: {self.entail_idx}")

    def _get_entailment_prob(self, premise, hypothesis):
        """计算单条 premise-hypothesis 对的 entailment 概率。"""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        return float(probs[self.entail_idx])

    def score_levels(self, pragmatic_text):
        """对 pragmatic_inference 文本，计算其与6个等级假设的 entailment 概率。"""
        scores = {}
        for lv, hypotheses in NLI_LEVEL_HYPOTHESES.items():
            entail_probs = []
            for hyp in hypotheses:
                p = self._get_entailment_prob(pragmatic_text, hyp)
                entail_probs.append(p)
            scores[lv] = float(np.mean(entail_probs))
        return scores


class LLMAnnotator:
    """
    基于 LLM 的自杀风险等级标注器。
    输入原始文本、语义改写、风险提示以及 IPTS/NLI 辅助评分，
    输出六级风险等级及从 IPTS 和 C-SSRS 双视角的分析依据。
    """

    def __init__(self, api_key=None, model="deepseek-chat",
                 base_url="https://api.deepseek.com"):
        from openai import OpenAI
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "请在 .env 文件中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，"
                "或通过 --api_key 参数传入"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = self._build_system_prompt()
        print(f"[LLMAnnotator] 模型: {model}")

    # ------------------------------------------------------------------ #
    #  系统提示词：嵌入完整 C-SSRS / IPTS 编码手册
    # ------------------------------------------------------------------ #
    def _build_system_prompt(self):
        cssrs_section = ""
        for lv in sorted(CSSRS_LEVEL_ARCHETYPES.keys()):
            info = CSSRS_LEVEL_ARCHETYPES[lv]
            cssrs_section += f"  等级{lv} [{info['label']}]: {info['description']}\n"

        ipts_section = ""
        for key, info in IPTS_ARCHETYPES.items():
            ipts_section += (
                f"  {key}（{info['name']}）: {info['description']}\n"
                f"    原型语句示例: {'; '.join(info['prototypes'][:3])}\n"
            )

        return (
            "你是一位专业的自杀风险评估专家，精通C-SSRS（哥伦比亚自杀严重程度评定量表）"
            "和IPTS（人际-心理理论）。\n"
            "你的任务是：根据提供的文本内容及辅助评分信息，判定该文本对应的自杀风险等级"
            "（0-5），并从 IPTS 和 C-SSRS 两个理论视角给出分析依据。\n\n"
            "【C-SSRS 六级风险等级标准】\n"
            f"{cssrs_section}\n"
            "【IPTS 八维构念定义及原型语句】\n"
            f"{ipts_section}\n"
            "【标注规则】\n"
            "1. 以「原始文本」和「语义改写」的实际内容为主要判断依据；\n"
            "   「IPTS维度相似度」和「NLI各等级蕴含概率」来自自动化模型，仅作参考，\n"
            "   当其与文本内容矛盾时以文本为准。\n"
            "2. 分析时先从 IPTS 视角识别文本涉及的心理构念维度及强度，\n"
            "   再从 C-SSRS 视角判定意念层级与行为层级，综合得出风险等级。\n"
            "3. 严格按以下 JSON 格式输出，不要添加任何多余文字：\n"
            "{\n"
            '  "risk_level": <int, 0-5>,\n'
            '  "explanation": "<str, 包含【IPTS分析】和【C-SSRS分析】两个段落>"\n'
            "}\n"
        )

    # ------------------------------------------------------------------ #
    #  单条标注
    # ------------------------------------------------------------------ #
    def annotate(self, original_text, literal_meaning, risk_hint,
                 ipts_scores, nli_scores, max_retries=3):
        """
        调用 LLM 对单条文本进行标注。
        返回 dict: {risk_level: int, explanation: str}
        """
        ipts_str = ", ".join(
            [f"{k}={v:.3f}" for k, v in ipts_scores.items()]
        )
        nli_str = ", ".join(
            [f"L{k}={v:.3f}" for k, v in nli_scores.items()]
        )

        user_prompt = (
            "请对以下文本进行自杀风险等级评估。\n\n"
            f"【原始文本】\n{original_text}\n\n"
            f"【语义改写】\n{literal_meaning}\n\n"
            f"【风险提示】\n{risk_hint if risk_hint else '无'}\n\n"
            f"【辅助参考 - IPTS维度相似度（0-1，越高越相关）】\n{ipts_str}\n\n"
            f"【辅助参考 - NLI各等级蕴含概率（0-1）】\n{nli_str}\n\n"
            "请严格按JSON格式输出，包含 risk_level (整数0-5) 和 explanation (字符串) 两个字段。"
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                result = self._parse_response(content)
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  [重试 {attempt+1}/{max_retries}] {e}，"
                          f"{wait}s 后重试")
                    time.sleep(wait)
                else:
                    print(f"  [标注失败] {e}")
                    return {"risk_level": -1,
                            "explanation": f"标注失败: {e}"}

    # ------------------------------------------------------------------ #
    #  解析 LLM 返回
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(content):
        """从 LLM 返回中提取 JSON，兼容 ```json...``` 包裹。"""
        # 去除可能的 markdown 代码块包裹
        cleaned = re.sub(r"^```(?:json)?\s*", "", content)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        result = json.loads(cleaned)
        level = int(result.get("risk_level", -1))
        if level < 0 or level > 5:
            raise ValueError(f"risk_level={level} 不在 0-5 范围内")
        return {
            "risk_level": level,
            "explanation": str(result.get("explanation", "")),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="IPTS + NLI + LLM 自杀风险等级标注"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入JSON文件路径 (final_semantic_completion.json)"
    )
    parser.add_argument(
        "--output", type=str, default="annotated_risk_dataset.json",
        help="输出JSON文件路径 (默认: annotated_risk_dataset.json)"
    )
    parser.add_argument(
        "--embedding_model", type=str,
        default="shibing624/text2vec-base-chinese",
        help="句向量模型名称 (默认: shibing624/text2vec-base-chinese)"
    )
    parser.add_argument(
        "--nli_model", type=str,
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        help="NLI模型名称"
    )
    parser.add_argument(
        "--llm_model", type=str, default="deepseek-chat",
        help="LLM模型名称 (默认: deepseek-chat)"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="DeepSeek API Key (也可通过 DEEPSEEK_API_KEY 环境变量设置)"
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.deepseek.com",
        help="LLM API Base URL (默认: https://api.deepseek.com)"
    )
    parser.add_argument(
        "--disable_nli", action="store_true",
        help="禁用NLI通道"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载数据 ----
    print(f"\n{'='*60}")
    print(f"IPTS + NLI + LLM 自杀风险等级标注")
    print(f"{'='*60}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[数据] 加载 {len(data)} 条记录")

    # ---- 初始化评分器 ----
    scorer = ArchetypeScorer(model_name=args.embedding_model)

    nli_scorer = None
    if not args.disable_nli:
        nli_scorer = NLIScorer(model_name=args.nli_model)
    else:
        print("[NLI] 已禁用NLI通道")

    # ---- 初始化 LLM 标注器 ----
    llm = LLMAnnotator(
        api_key=args.api_key,
        model=args.llm_model,
        base_url=args.base_url,
    )

    # ---- 标注主循环 ----
    results = []
    level_distribution = {i: 0 for i in range(6)}
    fail_count = 0

    print(f"\n[标注] 开始标注 {len(data)} 条数据...\n")

    for item in tqdm(data, desc="标注进度"):
        # ---------- 提取辅助评分 ----------
        literal = item.get("literal_meaning", "")
        pragmatic = item.get("pragmatic_inference", "")

        # IPTS 维度相似度 (基于 literal_meaning)
        if literal:
            literal_emb = scorer.encode_text(literal)
            ipts_scores = scorer.score_ipts(literal_emb)
        else:
            ipts_scores = {k: 0.0 for k in scorer.ipts_keys}

        # NLI 各等级蕴含概率 (基于 pragmatic_inference)
        if nli_scorer is not None and pragmatic:
            nli_scores = nli_scorer.score_levels(pragmatic)
        else:
            nli_scores = {lv: 0.0 for lv in range(6)}

        # ---------- LLM 标注 ----------
        original_text = item.get("content", "")

        # 从多个字段组合风险提示
        hint_parts = []
        if item.get("suicidal_ideation"):
            hint_parts.append(f"自杀意念类型: {item['suicidal_ideation']}")
        if item.get("behavioral_evidence"):
            hint_parts.append(f"行为证据: {item['behavioral_evidence']}")
        if item.get("emotional_intensity"):
            hint_parts.append(f"情绪强度: {item['emotional_intensity']}")
        if item.get("evidence_used"):
            hint_parts.append(
                f"关键隐语/证据: {', '.join(item['evidence_used'])}"
            )
        risk_hint = "; ".join(hint_parts) if hint_parts else ""

        llm_result = llm.annotate(
            original_text=original_text,
            literal_meaning=literal,
            risk_hint=risk_hint,
            ipts_scores=ipts_scores,
            nli_scores=nli_scores,
        )

        predicted_level = llm_result["risk_level"]
        explanation = llm_result["explanation"]

        # ---------- 组装结果 ----------
        result = dict(item)
        if predicted_level < 0:
            # 标注失败，保留原 risk_level 或置 -1
            fail_count += 1
            result["risk_level"] = item.get("risk_level", -1)
            result["risk_label"] = "标注失败"
            result["explanation"] = explanation
        else:
            result["risk_level"] = predicted_level
            result["risk_label"] = CSSRS_LEVEL_ARCHETYPES[predicted_level]["label"]
            result["explanation"] = explanation
            level_distribution[predicted_level] += 1

        result["annotation_method"] = "llm"
        result["scores"] = {
            "ipts_literal": {k: round(v, 4) for k, v in ipts_scores.items()},
            "level_nli": {str(k): round(v, 4) for k, v in nli_scores.items()},
        }
        results.append(result)

    # ---- 统计与输出 ----
    success_count = len(results) - fail_count
    print(f"\n{'='*60}")
    print(f"标注完成! 共 {len(results)} 条, 成功 {success_count}, 失败 {fail_count}")
    print(f"{'='*60}")
    print(f"等级分布 (仅成功标注):")
    for lv in range(6):
        count = level_distribution[lv]
        pct = count / success_count * 100 if success_count else 0
        bar = "█" * int(pct / 2)
        label = CSSRS_LEVEL_ARCHETYPES[lv]["label"]
        print(f"  等级{lv} ({label}): {count:>5} ({pct:>5.1f}%) {bar}")

    # 保存结果
    output_data = {
        "metadata": {
            "total_samples": len(results),
            "success_count": success_count,
            "fail_count": fail_count,
            "annotation_method": "llm_with_ipts_nli",
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "nli_model": args.nli_model if nli_scorer else None,
            "level_distribution": level_distribution,
            "timestamp": datetime.now().isoformat(),
        },
        "data": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[输出] 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
