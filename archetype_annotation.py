import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

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

# CSSRS_LEVEL_ARCHETYPES = {
#     0: {
#         "label": "无自杀意念",
#         "description": "日常负面情绪，未达到自杀风险层面",
#         "prototypes": [
#             "今天心情不太好，工作压力好大",
#             "和朋友吵架了好烦躁",
#             "失眠好几天了整个人很疲惫",
#             "最近状态不好什么都不想做",
#             "生活好无聊每天都一样",
#             "考试挂科了心情很差",
#             "被领导骂了一顿好郁闷",
#             "感觉自己很焦虑什么都做不好",
#             "每天加班累得不行好想休息",
#             "和男朋友分手了好难过好伤心",
#         ]
#     },
#     1: {
#         "label": "希望死亡但无计划",
#         "description": "表达了希望死亡、不想活等消极意愿，但无具体方法",
#         "prototypes": [
#             "不想活了活着好累",
#             "活着没有任何意义",
#             "如果能消失就好了",
#             "好想从这个世界上消失",
#             "活着太痛苦了不如死了算了",
#             "每天都不想醒来希望就这样睡过去",
#             "感觉活着就是在受罪",
#             "如果有一个按钮可以让我不存在就好了",
#             "生而为人我很抱歉不想继续了",
#             "真的好想解脱好想一切都结束",
#         ]
#     },
#     2: {
#         "label": "有自杀想法但无具体计划",
#         "description": "出现了自杀念头或暗示，但没有具体计划",
#         "prototypes": [
#             "脑海里反复出现想死的念头",
#             "有时候真的很想一了百了",
#             "想过要结束自己的生命",
#             "时不时会冒出去死的想法",
#             "越来越频繁地想到死亡",
#             "觉得死了就不用承受这些了",
#             "一直在想要不要结束这一切",
#             "不知道为什么脑子里总是想着去死",
#             "有那么几个瞬间真的很想死掉",
#             "想重开想离开这个世界",
#         ]
#     },
#     3: {
#         "label": "有自杀计划但未实施",
#         "description": "提及具体方法、时间安排或准备行为",
#         "prototypes": [
#             "我已经想好了要怎么做",
#             "打算用吃药的方式离开",
#             "已经选好了地点就在那栋楼",
#             "我开始写遗书了把想说的都写下来",
#             "把手机里该删的都删了该交代的都交代了",
#             "给自己定了一个日期那天就走",
#             "准备好了所有需要的东西",
#             "已经计划好了具体的时间和方式",
#             "这几天在做最后的安排",
#             "倒计时开始了还有几天",
#         ]
#     },
#     4: {
#         "label": "有自杀未遂或自伤行为",
#         "description": "描述了实际的自伤行为或自杀未遂经历",
#         "prototypes": [
#             "昨天又割了好几刀手臂上全是伤口",
#             "吃了一整瓶药被送去医院洗胃了",
#             "从楼上跳下来但是没死成",
#             "手腕上的疤一道又一道根本藏不住",
#             "上次od之后在icu躺了三天",
#             "割得太深了血止不住",
#             "每次情绪崩溃都会伤害自己",
#             "已经自杀未遂好几次了",
#             "吞了很多药醒来发现自己在医院",
#             "身上到处都是自残的痕迹",
#         ]
#     },
#     5: {
#         "label": "高致死性自杀企图",
#         "description": "高致死性的自杀企图或迫在眉睫的自杀意图",
#         "prototypes": [
#             "这是我发的最后一条消息",
#             "马上就要走了和这个世界告别",
#             "这次一定要成功不会再失败了",
#             "已经站在天台上了风好大",
#             "药已经吃下去了感觉意识在模糊",
#             "再见了所有人这是最后的告别",
#             "已经没有退路了今晚就是最后一晚",
#             "绳子已经准备好了马上就解脱了",
#             "正在从高处往下看一切即将结束",
#             "这次选了最致命的方式不会有意外了",
#         ]
#     },
# }
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

# NLI_LEVEL_HYPOTHESES = {
#     0: [
#         "此人表达了普通的负面情绪，没有任何自杀或自伤的意愿",
#         "此人只是心情不好或感到压力，没有涉及死亡或自杀的想法",
#     ],
#     1: [
#         "此人表达了不想活或希望死亡的意愿，但没有提到具体的方式或计划",
#         "此人流露出厌世情绪或觉得活着没有意义，但没有具体的自杀想法",
#     ],
#     2: [
#         "此人出现了关于自杀的想法或念头，但没有形成具体的计划",
#         "此人有反复出现的自杀意念，但没有决定具体的方式或时间",
#     ],
#     3: [
#         "此人有明确的自杀想法，可能已经确定了自杀的方式或时间",
#         "此人已经开始为自杀做准备，有明确的实施方案",
#     ],
#     4: [
#         "此人有实际的自伤行为或自杀未遂的经历",
#         "此人已经实施过自伤或自杀行为，如割伤、服药过量等",
#     ],
#     5: [
#         "此人正在实施或将在近期实施自杀行为",
#         "此人处于迫在眉睫的自杀危险中，明确表示将在近期内结束生命",
#     ],
# }
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
    使用literal_meaning字段计算余弦相似度
    """

    def __init__(self, model_name="shibing624/text2vec-base-chinese"):
        from sentence_transformers import SentenceTransformer
        print(f"[ArchetypeScorer] 加载句向量模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._encode_prototypes()

    def _encode_prototypes(self):
        """预编码 IPTS 和 C-SSRS 原型语句向量"""
        self.ipts_keys = list(IPTS_ARCHETYPES.keys())
        self.ipts_proto_embeddings = {}
        for key in self.ipts_keys:
            protos = IPTS_ARCHETYPES[key]["prototypes"]
            self.ipts_proto_embeddings[key] = self.model.encode(
                protos, normalize_embeddings=True, show_progress_bar=False
            )

        self.level_keys = sorted(CSSRS_LEVEL_ARCHETYPES.keys())
        self.level_proto_embeddings = {}
        for lv in self.level_keys:
            protos = CSSRS_LEVEL_ARCHETYPES[lv]["prototypes"]
            self.level_proto_embeddings[lv] = self.model.encode(
                protos, normalize_embeddings=True, show_progress_bar=False
            )
        print(f"[ArchetypeScorer] 原型编码完成: "
              f"IPTS {len(self.ipts_keys)}维, "
              f"C-SSRS {len(self.level_keys)}级")

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

    def score_levels(self, text_embedding):
        """计算文本向量与6个C-SSRS等级原型的平均余弦相似度。"""
        scores = {}
        for lv in self.level_keys:
            proto_embs = self.level_proto_embeddings[lv]
            sims = np.dot(proto_embs, text_embedding)
            scores[lv] = float(np.mean(sims))
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

        # 确定设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # 确定标签到索引的映射
        # mDeBERTa-v3-base-xnli 的标签顺序: entailment, neutral, contradiction
        self.label2id = self.model.config.label2id
        self.entail_idx = self.label2id.get("entailment", 0)
        print(f"[NLIScorer] 模型加载完成, 设备: {self.device}, "
              f"entailment索引: {self.entail_idx}")

    def _get_entailment_prob(self, premise, hypothesis):
        """
        计算单条 premise-hypothesis 对的 entailment 概率。
        返回: float, P(entailment)
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        return float(probs[self.entail_idx])

    def score_levels(self, pragmatic_text):
        """
        对 pragmatic_inference 文本，计算其与6个等级假设的 entailment 概率。
        """
        scores = {}
        for lv, hypotheses in NLI_LEVEL_HYPOTHESES.items():
            entail_probs = []
            for hyp in hypotheses:
                p = self._get_entailment_prob(pragmatic_text, hyp)
                entail_probs.append(p)
            scores[lv] = float(np.mean(entail_probs))
        return scores


def extract_features(item, scorer, nli_scorer=None):
    """
    提取特征向量。

    特征维度 (23维):
      [0:8]   ipts_literal    — literal_meaning 与 8个IPTS维度的余弦相似度
      [8:14]  level_literal   — literal_meaning 与 6个C-SSRS等级原型的余弦相似度
      [14:20] level_nli       — pragmatic_inference 与 6个等级假设的NLI entailment概率
                                 (若无NLI评分器则填0)
      [20]    max_ipts        — ipts_literal 中的最大值
      [21]    ipts_std        — ipts_literal 的标准差 (分辨性指标)
      [22]    max_level_sim   — level_literal 中的最大值
    """
    literal = item.get("literal_meaning", "")
    pragmatic = item.get("pragmatic_inference", "")

    # 通道1: Archetype (基于 literal_meaning)
    literal_emb = scorer.encode_text(literal) if literal else np.zeros(768)
    ipts_scores = scorer.score_ipts(literal_emb)
    level_literal_scores = scorer.score_levels(literal_emb)

    # 通道2: NLI (基于 pragmatic_inference)
    if nli_scorer is not None and pragmatic:
        level_nli_scores = nli_scorer.score_levels(pragmatic)
    else:
        level_nli_scores = {lv: 0.0 for lv in range(6)}

    # 组装
    ipts_vec = [ipts_scores[k] for k in scorer.ipts_keys]          # 8维
    level_lit_vec = [level_literal_scores[lv] for lv in range(6)]   # 6维
    level_nli_vec = [level_nli_scores[lv] for lv in range(6)]       # 6维

    derived = [
        float(np.max(ipts_vec)),     # max_ipts
        float(np.std(ipts_vec)),     # ipts_std
        float(np.max(level_lit_vec)) # max_level_sim
    ]

    feature_vector = np.array(ipts_vec + level_lit_vec + level_nli_vec + derived,
                              dtype=np.float32)

    return {
        "feature_vector": feature_vector,           # 23维 ndarray
        "ipts_literal": ipts_scores,                # dict
        "level_literal": level_literal_scores,      # dict
        "level_nli": level_nli_scores,              # dict
        "max_ipts": derived[0],
        "ipts_std": derived[1],
        "max_level_sim": derived[2],
    }


def predict_by_similarity(feature_dict, archetype_weight=0.4, nli_weight=0.6):
    """
    路径A: 双通道加权投票预测风险等级。

    公式:
      combined[i] = archetype_weight × level_literal[i]
                  + nli_weight      × level_nli[i]
      predicted_level = argmax(combined)
    """
    level_lit = feature_dict["level_literal"]
    level_nli = feature_dict["level_nli"]

    combined = {}
    for lv in range(6):
        combined[lv] = (archetype_weight * level_lit[lv]
                        + nli_weight * level_nli[lv])

    # 排序取最高
    sorted_levels = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    predicted_level = sorted_levels[0][0]
    top_score = sorted_levels[0][1]
    second_score = sorted_levels[1][1] if len(sorted_levels) > 1 else 0.0
    confidence = top_score - second_score

    return predicted_level, confidence, {
        "combined_scores": combined,
        "level_literal": level_lit,
        "level_nli": level_nli,
        "archetype_weight": archetype_weight,
        "nli_weight": nli_weight,
    }

def train_calibration_classifier(calibration_data, scorer, nli_scorer=None):
    """
    路径B: 使用少量人工标注数据训练RandomForest分类器。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report

    print(f"[路径B] 使用 {len(calibration_data)} 条标注数据训练分类器...")

    X, y = [], []
    for item in tqdm(calibration_data, desc="提取校准特征"):
        feat = extract_features(item, scorer, nli_scorer)
        X.append(feat["feature_vector"])
        y.append(int(item["risk_level"]))

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 如果样本足够，做交叉验证评估
    if len(y) >= 20:
        n_splits = min(5, len(y) // 4)
        if n_splits >= 2:
            cv_scores = cross_val_score(clf, X, y, cv=n_splits, scoring="f1_macro")
            print(f"[路径B] {n_splits}-fold CV macro-F1: "
                  f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred, zero_division=0)
    print(f"[路径B] 训练集分类报告:\n{report}")

    return clf, report


def predict_by_classifier(feature_vector, classifier):
    """
    使用训练好的分类器预测风险等级。
    """
    X = feature_vector.reshape(1, -1)
    predicted_level = int(classifier.predict(X)[0])
    proba = classifier.predict_proba(X)[0]
    classes = classifier.classes_
    probabilities = {int(c): float(p) for c, p in zip(classes, proba)}
    confidence = probabilities.get(predicted_level, 0.0)

    return predicted_level, confidence, probabilities

def parse_args():
    parser = argparse.ArgumentParser(
        description="Archetype + NLI 双通道自杀风险等级标注"
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
        "--calibration", type=str, default=None,
        help="[路径B] 人工标注的校准数据JSON路径"
    )
    parser.add_argument(
        "--embedding_model", type=str,
        default="shibing624/text2vec-base-chinese",
        help="句向量模型名称 (默认: shibing624/text2vec-base-chinese)"
    )
    parser.add_argument(
        "--nli_model", type=str,
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        help="NLI模型名称 (默认: mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)"
    )
    parser.add_argument(
        "--archetype_weight", type=float, default=0.4,
        help="Archetype通道权重 (默认: 0.4)"
    )
    parser.add_argument(
        "--nli_weight", type=float, default=0.6,
        help="NLI通道权重 (默认: 0.6)"
    )
    parser.add_argument(
        "--disable_nli", action="store_true",
        help="禁用NLI通道，仅使用Archetype通道"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载数据 ----
    print(f"\n{'='*60}")
    print(f"Archetype + NLI 双通道自杀风险等级标注")
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
        print("[NLI] 已禁用NLI通道，仅使用Archetype通道")

    # ---- 路径B: 加载校准数据并训练分类器 ----
    classifier = None
    if args.calibration:
        with open(args.calibration, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        print(f"[路径B] 加载 {len(cal_data)} 条校准数据")
        classifier, _ = train_calibration_classifier(cal_data, scorer, nli_scorer)

    # ---- 标注主循环 ----
    results = []
    level_distribution = {i: 0 for i in range(6)}
    confidence_sum = 0.0

    # 确定权重 (若NLI被禁用，全部权重给Archetype)
    if nli_scorer is not None:
        w_arch = args.archetype_weight
        w_nli = args.nli_weight
    else:
        w_arch = 1.0
        w_nli = 0.0

    # 权重归一化
    w_total = w_arch + w_nli
    if w_total > 0:
        w_arch /= w_total
        w_nli /= w_total

    print(f"\n[标注] 通道权重: Archetype={w_arch:.2f}, NLI={w_nli:.2f}")
    print(f"[标注] 开始标注 {len(data)} 条数据...\n")

    for item in tqdm(data, desc="标注进度"):
        # 提取特征
        feat = extract_features(item, scorer, nli_scorer)

        # 预测
        if classifier is not None:
            predicted_level, confidence, proba = predict_by_classifier(
                feat["feature_vector"], classifier
            )
            method = "classifier"
        else:
            predicted_level, confidence, details = predict_by_similarity(
                feat, archetype_weight=w_arch, nli_weight=w_nli
            )
            method = "dual_channel"

        # 组装结果
        result = dict(item)  # 保留原始字段
        result["risk_level"] = predicted_level
        result["risk_label"] = CSSRS_LEVEL_ARCHETYPES[predicted_level]["label"]
        result["confidence"] = round(confidence, 4)
        result["annotation_method"] = method

        # 保存详细分数 (便于后续分析和调试)
        result["scores"] = {
            "ipts_literal": feat["ipts_literal"],
            "level_literal": {str(k): round(v, 4) for k, v in feat["level_literal"].items()},
            "level_nli": {str(k): round(v, 4) for k, v in feat["level_nli"].items()},
            "max_ipts": round(feat["max_ipts"], 4),
            "ipts_std": round(feat["ipts_std"], 4),
        }

        if classifier is None:
            result["scores"]["combined"] = {
                str(k): round(v, 4) for k, v in details["combined_scores"].items()
            }

        results.append(result)
        level_distribution[predicted_level] += 1
        confidence_sum += confidence

    # ---- 统计与输出 ----
    avg_confidence = confidence_sum / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print(f"标注完成! 共 {len(results)} 条")
    print(f"{'='*60}")
    print(f"平均置信度: {avg_confidence:.4f}")
    print(f"等级分布:")
    for lv in range(6):
        count = level_distribution[lv]
        pct = count / len(results) * 100 if results else 0
        bar = "█" * int(pct / 2)
        label = CSSRS_LEVEL_ARCHETYPES[lv]["label"]
        print(f"  等级{lv} ({label}): {count:>5} ({pct:>5.1f}%) {bar}")

    # 保存结果
    output_data = {
        "metadata": {
            "total_samples": len(results),
            "annotation_method": "dual_channel_archetype_nli"
                                 if nli_scorer else "archetype_only",
            "archetype_weight": w_arch,
            "nli_weight": w_nli,
            "embedding_model": args.embedding_model,
            "nli_model": args.nli_model if nli_scorer else None,
            "calibration_data": args.calibration,
            "level_distribution": level_distribution,
            "avg_confidence": round(avg_confidence, 4),
            "timestamp": datetime.now().isoformat(),
        },
        "data": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[输出] 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
