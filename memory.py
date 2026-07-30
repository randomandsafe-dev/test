"""对话记忆管理模块

基于 LangChain ConversationBufferMemory 实现对话上下文的持久化与重置。
"""
from __future__ import annotations

import logging

from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

# 记忆存储键名
_MEMORY_KEY = "chat_history"
_INPUT_KEY = "input"


def get_memory() -> ConversationBufferMemory:
    """创建并返回一个全新的对话记忆实例。

    记忆实例会保存完整的对话历史，供 LLMChain 使用。

    Returns:
        ConversationBufferMemory: 新记忆实例，output_key 保证 LLMChain 能正确对接。
    """
    logger.info("创建新的对话记忆实例 (memory_key=%s)", _MEMORY_KEY)
    return ConversationBufferMemory(
        memory_key=_MEMORY_KEY,
        input_key=_INPUT_KEY,
        return_messages=True,
    )


def clear_memory() -> ConversationBufferMemory:
    """清空当前对话历史并返回一个全新的记忆实例。

    与 get_memory() 行为完全一致——创建全新的 ConversationBufferMemory。
    调用方可直接将返回值赋给正在使用的 memory 变量，实现"重置"效果。

    Returns:
