# -*- coding: utf-8 -*-
"""
LettaAutoMemoryModel — Mastermind + Letta（兼容旧版 SDK + 工具/调用诊断）
- 不注入任何游戏规则/策略，只约束输出格式（零提示）。
- 可选：tools_hint（存在性提示）、force_tools（每轮必须写私有记忆）。
- 兼容旧版 letta_client：agents.create() 仅支持 model / embedding / system / memory_blocks。
- tool_choice 优先 "required" → 不支持降级到 "auto" → 错误时兜底到 "none"。
- 每轮打印 memory blocks（rules/state）、工具清单、工具调用意图。
- 不造默认输出；模型若无输出，就返回空字符串，便于真实评估。
"""

from typing import List, Dict, Optional, Any
from letta_client import Letta

ChatHistory = List[Dict[str, str]]

# 最小 I/O 约束：只限定输出格式，不包含任何规则/策略内容
BASE_INSTRUCTION = "Respond only with a single line formatted as: FINAL GUESS:['c1','c2','c3','c4']"


class LettaAutoMemoryModel:
    def __init__(
        self,
        base_url: str = "http://localhost:8283",
        model: str = "openai/gpt-4o-mini",
        embedding: str = "openai/text-embedding-3-small",
        token: Optional[str] = None,
        verbose: bool = True,
        # 旧 SDK 不支持：include_base_tools / enable_sleeptime / type 等
        state_limit: int = 6000,
        rules_limit: int = 8000,
        print_mem_each_round: bool = True,  # 每轮打印记忆
        tools_mode: str = "auto",           # "auto" 或 "none"
        tools_hint: bool = False,           # 仅提示“可用工具”（不含游戏信息）
        force_tools: bool = False,          # 强制每轮先用工具（若环境/模型遵循）
    ):
        self.verbose = verbose
        self.print_mem_each_round = print_mem_each_round
        self._tools_disabled = (tools_mode.lower() == "none")
        self._tools_hint = bool(tools_hint)
        self._force_tools = bool(force_tools)
        self._max_tail_messages = 1  # 只发最后一条 user 消息

        # 初始化 Letta 客户端（token 优先）
        self.client = Letta(token=token) if token else Letta(base_url=base_url)

        # 生成最小 system（零提示 / 存在性提示 / 强制使用工具）
        system_text = self._make_system_text()

        # 用旧 SDK 支持的最小参数创建 Agent
        create_kwargs = dict(
            model=model,
            embedding=embedding,
            system=system_text,
            memory_blocks=[
                {"label": "rules", "description": "Short text.", "limit": rules_limit},
                {"label": "state", "description": "Short text.", "limit": state_limit},
            ],
        )
        agent = self.client.agents.create(**create_kwargs)
        self.agent_id: Optional[str] = getattr(agent, "id", None) or (
            agent.get("id") if isinstance(agent, dict) else None
        )
        if not self.agent_id:
            raise RuntimeError("Letta agent creation failed: id is None")

        # 创建成功后立即打印工具清单（诊断）
        try:
            self._print_agent_tools()
        except Exception as e:
            if self.verbose:
                print("[Letta] tools introspection error:", e)

    # ----------------- 辅助方法 -----------------
    def _make_system_text(self) -> str:
        """根据开关生成最小 system 文本；不包含任何游戏知识或策略。"""
        if self._force_tools:
            # 强制要求：每轮必须先用可用工具更新私有记忆，再回复
            return (
                BASE_INSTRUCTION
                + " Before each reply, you MUST use available tools (e.g., memory_insert, memory_replace; or legacy core_memory_append, core_memory_replace) "
                  "to update your PRIVATE memory blocks with ONE short line describing the last turn. "
                  "NEVER include memory contents in your reply."
            )
        if self._tools_hint:
            # 仅存在性提示：可用工具维护私有记忆
            return (
                BASE_INSTRUCTION
                + " You may use available tools (e.g., memory_insert, memory_replace; or legacy core_memory_append, core_memory_replace) "
                  "to manage your PRIVATE memory if beneficial; never include memory contents in your reply."
            )
        return BASE_INSTRUCTION

    def _last_user_only(self, chat_history: ChatHistory) -> ChatHistory:
        """只取最后一条用户消息；若没有，则给极简提示。"""
        for m in reversed(chat_history):
            if (m.get("role") or "").lower() == "user":
                c = m.get("content") or ""
                return [{"role": "user", "content": c if isinstance(c, str) else str(c)}]
        return [{"role": "user", "content": "Please output your next guess only as FINAL GUESS:[...]"}]

    def _extract_last_text(self, resp: Any) -> str:
        """兼容不同 SDK 结构，取出最后一条文本。"""
        msgs = getattr(resp, "messages", None) or (
            resp.get("messages") if isinstance(resp, dict) else None
        )
        if not isinstance(msgs, list) or not msgs:
            return ""
        last = msgs[-1]
        content = getattr(last, "content", None) if hasattr(last, "content") else last.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        return t.strip()
        return getattr(last, "text", None) if hasattr(last, "text") else last.get("text", "")

    def _print_memory_blocks(self):
        """逐 label 精确读取并打印（rules/state）。"""
        try:
            def get_block(label: str) -> str:
                try:
                    blk = self.client.agents.blocks.retrieve(agent_id=self.agent_id, block_label=label)
                    return (
                        getattr(blk, "content", None) or getattr(blk, "value", None) or
                        (blk.get("content") if isinstance(blk, dict) else None) or
                        (blk.get("value") if isinstance(blk, dict) else None) or
                        ""
                    )
                except Exception:
                    return ""

            rules_txt = get_block("rules")
            state_txt = get_block("state")

            print("[Letta][MemoryBlocks]")
            print(f"- rules:\n{(rules_txt or '<empty>').strip()}")
            print(f"- state:\n{(state_txt or '<empty>').strip()}")
        except Exception as e:
            print(f"[Letta][MemoryBlocks][Error] {e}")

    def _print_agent_tools(self):
        """列出当前 Agent 的工具清单（诊断）。兼容不同 SDK 返回结构。"""
        print("[Letta][Tools]")
        tools = []
        try:
            # 新一点的 SDK 可能有 agents.tools.list
            if hasattr(self.client.agents, "tools") and hasattr(self.client.agents.tools, "list"):
                res = self.client.agents.tools.list(self.agent_id)
                maybe = getattr(res, "tools", None) or getattr(res, "data", None) or res
                if isinstance(maybe, list):
                    tools = maybe
            else:
                # 老 SDK：尝试从 agent 对象取
                agent = self.client.agents.retrieve(self.agent_id)
                maybe = getattr(agent, "tools", None) or (agent.get("tools") if isinstance(agent, dict) else None)
                if isinstance(maybe, list):
                    tools = maybe
        except Exception as e:
            print(f"[Letta][Tools][Error] {e}")

        if not tools:
            print("- <no-tools-visible>")
            print(" 没看到任何工具。若希望“纯 Letta 自主写记忆”，请确认服务端启用默认记忆工具，或升级 letta_server/letta_client。")
            return

        names = []
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name") or t.get("tool") or t.get("id") or str(t)
            else:
                name = getattr(t, "name", None) or getattr(t, "tool", None) or getattr(t, "id", None) or str(t)
            names.append(name)

        for n in names:
            print(f"- {n}")

        mem_new = any(n in ("memory_insert", "memory_replace") for n in names)
        mem_old = any(n in ("core_memory_append", "core_memory_replace") for n in names)
        if not (mem_new or mem_old):
            print("未发现记忆工具（memory_insert/memory_replace 或 core_memory_*）。Agent 无法“自主写记忆”。")
        else:
            print(" 发现记忆相关工具（新或旧）。若仍然不写，多半是模型没有触发工具调用。")

    def _print_tool_calls(self, resp_obj: Any):
        """尽量兼容地打印是否检测到“工具调用意图”（tool calls / actions）。"""
        def _get(obj, key):
            return getattr(obj, key, None) if hasattr(obj, key) else (obj.get(key) if isinstance(obj, dict) else None)

        calls = []
        # 位点1：顶层 actions / tool_calls
        for key in ("actions", "tool_calls"):
            v = _get(resp_obj, key)
            if isinstance(v, list) and v:
                calls = v
                break
        # 位点2：最后一条消息的 actions / tool_calls
        if not calls:
            msgs = _get(resp_obj, "messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                for key in ("actions", "tool_calls"):
                    v = _get(last, key)
                    if isinstance(v, list) and v:
                        calls = v
                        break

        print("[Letta][ToolCalls]")
        if not calls:
            print("- <no-tool-calls-detected>")
            return
        for c in calls:
            if isinstance(c, dict):
                name = c.get("name") or c.get("tool_name") or c.get("tool") or "<unnamed>"
            else:
                name = getattr(c, "name", None) or getattr(c, "tool_name", None) or getattr(c, "tool", None) or str(c)
            print(f"- {name}")

    # ----------------- 主调用 -----------------
    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        # 仅发最后一条 user，避免上下文膨胀
        tail = self._last_user_only(chat_history)

        # 选择工具策略
        if self._tools_disabled:
            use_tool_choice = "none"
        else:
            use_tool_choice = "required" if self._force_tools else "auto"

        try:
            # 优先尝试带 tool_choice；若 SDK 不支持，则降级
            try:
                resp = self.client.agents.messages.create(
                    agent_id=self.agent_id,
                    messages=tail,
                    tool_choice=use_tool_choice,
                )
            except TypeError:
                # 旧 SDK 不支持 tool_choice 参数或不支持 "required"：退回不传（≈ auto）
                resp = self.client.agents.messages.create(agent_id=self.agent_id, messages=tail)

            content = self._extract_last_text(resp)
            if self.verbose:
                print("[Letta] Raw reply:", content if content else "<EMPTY>")

            # 打印工具调用意图（诊断）
            try:
                self._print_tool_calls(resp)
            except Exception as e:
                if self.verbose:
                    print("[Letta] tool-call introspection error:", e)

            # 打印记忆块（诊断）
            if self.print_mem_each_round:
                self._print_memory_blocks()

            # 不造默认输出；如果模型没回内容，就返回空字符串
            if not content and self.verbose:
                print("[Letta] Empty reply (no output from model)")
            return chat_history + [{"role": "assistant", "content": content}]

        except Exception as e:
            if self.verbose:
                print("[Letta] error:", e)

            # 若还未禁用工具，则降级为不用工具再试一次（确保整局不中断）
            if not self._tools_disabled:
                self._tools_disabled = True
                if self.verbose:
                    print("[Letta] tool path failed; fallback to none")
                try:
                    resp = self.client.agents.messages.create(
                        agent_id=self.agent_id,
                        messages=tail,
                        tool_choice="none",
                    )
                    content = self._extract_last_text(resp)
                    if self.verbose:
                        print("[Letta] Raw reply (fallback):", content if content else "<EMPTY>")
                    try:
                        self._print_tool_calls(resp)
                    except Exception as e2:
                        if self.verbose:
                            print("[Letta] tool-call introspection error (fallback):", e2)
                    if self.print_mem_each_round:
                        self._print_memory_blocks()
                    if not content and self.verbose:
                        print("[Letta] Empty reply (fallback, no output)")
                    return chat_history + [{"role": "assistant", "content": content}]
                except Exception as e2:
                    if self.verbose:
                        print("[Letta] fallback error:", e2)

            # 最终错误：返回空字符串，方便你识别“无输出”
            return chat_history + [{"role": "assistant", "content": ""}]

    def get_model_info(self) -> str:
        mode = "no-tools" if self._tools_disabled else ("forced-tools" if self._force_tools else "auto-tools")
        hint = "with-tools-hint" if self._tools_hint else "zero-prompt"
        return f"Letta Auto-Memory Model ({hint}, {mode})"
