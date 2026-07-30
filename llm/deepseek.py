"""DeepSeek LLM 适配器

通过 ChatOpenAI 兼容接口接入 DeepSeek，提供统一的工厂函数。
DeepSeek API 与 OpenAI SDK 完全兼容，仅需替换 base_url 和 api_key。
"""
from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from config import get_settings

logger = logging.getLogger(__name__)


def get_deepseek_llm() -> ChatOpenAI:
    """创建并返回配置好的 DeepSeek ChatOpenAI 实例。

    从 config.Settings 读取所有连接和模型参数。

    Returns:
        已配置的 ChatOpenAI 实例，指向 DeepSeek API。

    Raises:
        ValueError: 当 API 密钥未配置时。
    """
    settings = get_settings()

    logger.info(
        "初始化 DeepSeek LLM: model=%s, base_url=%s, temperature=%.2f",
        settings.deepseek_model,
        settings.deepseek_base_url,
        settings.deepseek_temperature,
    )

    return ChatOpenAI(
        model=settings.deepseek_model,
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=settings.deepseek_base_url,
