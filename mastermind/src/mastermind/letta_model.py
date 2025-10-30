# src/mastermind/letta_model.py
from typing import List, Dict, Optional
from letta_client import Letta
from mastermind.utils import parse_guess
import re

ChatHistory = List[Dict[str, str]]

class LettaModel:
    def __init__(
        self,
        base_url: str = "http://localhost:8283",
        model: str = "openai/gpt-4o-mini",
        embedding: str = "openai/text-embedding-3-small",
        token: Optional[str] = None,
        verbose: bool = False,
        print_mem_each_round: bool = True,   # 每轮是否打印记忆
        show_rules_in_debug: bool = False,   # debug 是否显示 rules（默认不显示）
        show_last_n_rounds: int = 12,        # 仅展示最近 N 轮
    ):
        self.client = Letta(token=token) if token else Letta(base_url=base_url)
        self.model_name = model
        self.embedding = embedding
        self.verbose = verbose
        self.print_mem_each_round = print_mem_each_round
        self.show_rules_in_debug = show_rules_in_debug
        self.show_last_n_rounds = max(1, int(show_last_n_rounds))

        # 只保留 rules / state 两块
        agent = self.client.agents.create(
            model=model,
            embedding=embedding,
            memory_blocks=[
                {"label": "rules", "value": ""},  # 首轮写一次
                {"label": "state", "value": ""},  # 累积：Round N: guess=...; feedback: ...
            ],
        )
        self.agent_id = getattr(agent, "id", None)
        if not self.agent_id:
            raise RuntimeError("Letta agent creation failed")

        self._bootstrap = "Please make a guess."
        self._round = 0  # 轮次计数

    # ========= 基础工具 =========
    def _last_user_only(self, chat_history: ChatHistory) -> ChatHistory:
        # 只取最近一条 user；没有就给一个最简提示
        for m in reversed(chat_history):
            if (m.get("role") or "").lower() == "user":
                c = m.get("content", "")
                if not isinstance(c, str):
                    c = str(c)
                c = c.strip()
                if c:
                    return [{"role": "user", "content": c}]
        return [{"role": "user", "content": self._bootstrap}]

    def _extract_last_text(self, resp) -> str:
        msgs = getattr(resp, "messages", None)
        if msgs is None and isinstance(resp, dict):
            msgs = resp.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return ""
        def pick(m) -> str:
            content = getattr(m, "content", None) if hasattr(m, "content") else m.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict):
                        t = p.get("text")
                        if isinstance(t, str) and t.strip():
                            return t.strip()
            text = getattr(m, "text", None) if hasattr(m, "text") else m.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            return ""
        for m in reversed(msgs):
            t = pick(m)
            if t:
                return t
        return ""

    # ========= 规则与反馈抽取 =========
    def _extract_rules_text(self, text: str) -> str:
        """从首轮 user 文本中提取规则（命中关键词就原样存）。"""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        if re.search(r"\b(Your goal is to guess|The game is defined|Allowed colors|Code length|duplicates)\b", t, re.I):
            return t
        return ""

    def _extract_feedback(self, text: str) -> Dict[str, str]:
        """
        检测真实反馈：
        - 明确 'Feedback:' 段，或包含 'Correct color and position' / 'wrong position' 等评分
        - 明显是规则长文时视为无反馈
        """
        if not isinstance(text, str):
            return {"has_feedback": False}
        t = text.strip()

        looks_like_rules = bool(re.search(r"\b(Your goal is to guess|The game is defined)\b", t, re.I))
        m = re.search(r"Feedback\s*:(.*)$", t, flags=re.I | re.S)
        segment = m.group(1).strip() if m else t

        exact = re.search(r"(?:Correct color and position|exact[^:]*):\s*(\d+)", segment, flags=re.I)
        color_only = re.search(r"(?:Correct color but wrong position|wrong position[^:]*):\s*(\d+)", segment, flags=re.I)

        has = bool(m or exact or color_only)
        if not has and looks_like_rules:
            return {"exact": "", "color_only": "", "raw": "", "has_feedback": False}

        return {
            "exact": exact.group(1) if exact else "",
            "color_only": color_only.group(1) if color_only else "",
            "raw": ("Feedback: " + segment) if m else (segment if has else ""),
            "has_feedback": has,
        }

    # ========= Memory Blocks 基础 =========
    def _set_block_value(self, label: str, value: str):
        if not label:
            return
        # 优先使用 block_label 修改
        try:
            self.client.agents.blocks.modify(
                agent_id=self.agent_id,
                block_label=label,
                value=value or ""
            )
            return
        except Exception:
            pass
        # 兼容：通过 block_id 修改
        try:
            blk = self.client.agents.blocks.retrieve(agent_id=self.agent_id, block_label=label)
            block_id = getattr(blk, "id", None) or (blk.get("id") if isinstance(blk, dict) else None)
            if block_id:
                try:
                    self.client.agents.blocks.modify(
                        agent_id=self.agent_id,
                        block_id=block_id,
                        value=value or ""
                    )
                    return
                except Exception:
                    pass
            if block_id and hasattr(self.client, "blocks"):
                self.client.blocks.modify(block_id=block_id, value=value or "")
        except Exception as e:
            if self.verbose:
                print(f"[Letta] modify block '{label}' error:", e)

    def _retrieve_block_value(self, label: str) -> str:
        try:
            blk = self.client.agents.blocks.retrieve(agent_id=self.agent_id, block_label=label)
            return getattr(blk, "value", None) or (blk.get("value") if isinstance(blk, dict) else "") or ""
        except Exception:
            return ""

    # ========= 写入 rules（只做一次） =========
    def _ensure_rules_once(self, source_text: str):
        try:
            existing = self._retrieve_block_value("rules")
            if existing:
                return
            rules_text = self._extract_rules_text(source_text)
            if rules_text:
                self._set_block_value("rules", rules_text)
                if self.verbose:
                    print("[Letta][Rules set]")
        except Exception as e:
            if self.verbose:
                print("[Letta][Rules set error]:", e)

    # ========= 批量 upsert 写入 state（支持静默写） =========
    def _batch_upsert_state(self, updates, print_now: bool = True):
        """
        批量 upsert 到 state；updates 是一组 (round_no, guess, feedback_dict_or_None) 元组。
        print_now: 是否立即打印 Memory updated 和 MemoryBlocks
        行格式：Round N: guess=[...]; feedback: exact=?, color-only=? / N/A
        """
        old = self._retrieve_block_value("state")

        # 解析旧内容 -> {round_no: {'guess': str|None, 'fb': str|None}}
        by_round = {}
        if old:
            for ln in old.splitlines():
                m = re.match(r"\s*Round\s+(\d+)\s*:\s*guess=(.+?);\s*feedback:\s*(.*)\s*$", ln)
                if m:
                    r = int(m.group(1))
                    by_round[r] = {"guess": m.group(2).strip(), "fb": m.group(3).strip()}
                else:
                    m2 = re.match(r"\s*Round\s+(\d+)\s*:\s*(.*)$", ln)
                    if m2:
                        r = int(m2.group(1))
                        by_round[r] = {"guess": m2.group(2).strip(), "fb": None}

        # 应用批量更新
        for (rno, g, fb) in updates:
            cur = by_round.get(rno, {"guess": None, "fb": None})
            if g is not None:
                cur["guess"] = g
            if fb is not None:
                if fb.get("has_feedback"):
                    cur["fb"] = f"exact={fb.get('exact','')}, color-only={fb.get('color_only','')}"
                elif cur["fb"] is None:
                    cur["fb"] = "N/A"
            by_round[rno] = cur

        # 按轮次升序重建文本（保证顺序稳定）
        rebuilt = []
        for r in sorted(by_round.keys()):
            item = by_round[r]
            gtxt = item["guess"] or ""
            ftxt = item["fb"] if item["fb"] is not None else "N/A"
            rebuilt.append(f"Round {r}: guess={gtxt}; feedback: {ftxt}")
        new_val = "\n".join(rebuilt).strip()

        if new_val != old:
            self._set_block_value("state", new_val)
            if print_now and self.verbose:
                print("[Letta][Memory updated]")
                if self.print_mem_each_round:
                    self.debug_print_memory()

    # ========= 调试查看 =========
    def _tail_state_lines(self, text: str, n: int) -> str:
        """只返回末尾 n 行（最近 n 轮）。"""
        if not text:
            return ""
        lines = text.splitlines()
        return "\n".join(lines[-n:])

    def debug_print_memory(self):
        """默认只打印 state 的最近 N 行；如需看 rules，把 show_rules_in_debug=True。"""
        print("[Letta][MemoryBlocks]")
        # state（最近 N 轮）
        try:
            v = self._retrieve_block_value("state")
            print(f"- state:\n{self._tail_state_lines(v, self.show_last_n_rounds)}\n")
        except Exception as e:
            print(f"- state: <retrieve error: {e}>")

        # rules（可选显示）
        if self.show_rules_in_debug:
            try:
                r = self._retrieve_block_value("rules")
                print(f"- rules:\n{r}\n")
            except Exception as e:
                print(f"- rules: <retrieve error: {e}>")

    # ========= 主调用 =========
    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        self._round += 1

        # 首轮尝试写入 rules（只做一次）
        if self._round == 1:
            # 优先找第一条 user，否则退回到最近一条
            first_user = ""
            for m in chat_history:
                if (m.get("role") or "").lower() == "user":
                    first_user = m.get("content") or ""
                    break
            if not first_user:
                last_only = self._last_user_only(chat_history)
                first_user = last_only[0]["content"]
            self._ensure_rules_once(first_user)

        # 当前这条 user（通常包含上一轮反馈）
        msgs = self._last_user_only(chat_history)
        last_user_txt = msgs[0]["content"]
        feedback = self._extract_feedback(last_user_txt)

        # —— 先把上一轮(N-1)反馈写入（静默，不打印），这样如果 Letta 会读取 memory，本轮能用到上一轮信息
        if self._round > 1 and feedback.get("has_feedback"):
            self._batch_upsert_state([(self._round - 1, None, feedback)], print_now=False)

        # 请求模型，拿到本轮猜测
        content = ""
        try:
            resp = self.client.agents.messages.create(agent_id=self.agent_id, messages=msgs)
            content = self._extract_last_text(resp)
            if not content:
                raise RuntimeError("Empty content from Letta")
        except Exception as e:
            if self.verbose:
                print(f"[Letta] error: {e}")
            # 极简重试（不设默认猜测）
            try:
                resp = self.client.agents.messages.create(
                    agent_id=self.agent_id,
                    messages=[{"role": "user", "content": "Please reply with your next guess only."}]
                )
                content = self._extract_last_text(resp)
            except Exception as e2:
                if self.verbose:
                    print(f"[Letta] fallback error: {e2}")
                content = ""

        # 打印 + 解析猜测
        parsed_guess = None
        if self.verbose:
            print("[Letta] Raw reply:", content if content else "<EMPTY>")
        try:
            parsed_guess = parse_guess({"role": "assistant", "content": content})
            if self.verbose and parsed_guess:
                print("[Letta] Parsed guess:", parsed_guess)
        except Exception:
            pass

        # —— 再把本轮(N)的猜测写入（正常打印一次）
        round_guess = str(parsed_guess) if parsed_guess else content
        self._batch_upsert_state([(self._round, round_guess, None)], print_now=True)

        # 返回给主流程
        return chat_history + [{"role": "assistant", "content": content}]

    def get_model_info(self) -> str:
        return f"Letta Model: {self.model_name}"
