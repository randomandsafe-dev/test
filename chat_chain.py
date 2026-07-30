"""对话链模块

使用 LLMChain 串联 DeepSeek 模型与对话记忆，提供统一的对话调用接口。
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 系统提示词 — 定义助手的角色和行为
_SYSTEM_PROMPT = (
    "你是一个友好、乐于助人的 AI 助手。"
    "你总是用中文回答用户的问题，除非用户使用其他语言。"
    "你的回答清晰、准确、有礼貌。"
)

# 对话模板：系统消息 + 历史消息占位符 + 用户当前输入
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


def create_chat_chain(
    llm: ChatOpenAI, memory: ConversationBufferMemory
) -> LLMChain:
    """创建对话链，将 LLM 与记忆组件绑定。

    Args:
        llm: 配置好的 ChatOpenAI 实例（已指向 DeepSeek）。
        memory: ConversationBufferMemory 实例，用于保存和加载对话历史。

    Returns:
        LLMChain: 已配置好 prompt、llm 和 memory 的对话链。
    """
    logger.info("创建 LLMChain — 绑定 prompt + llm + memory")
    return LLMChain(
        llm=llm,
        prompt=CHAT_PROMPT,
        memory=memory,
        verbose=False,
    )


async def run_chain(chain: LLMChain, user_input: str) -> str:
    """执行对话链，传入用户输入并返回 AI 回复。

    Args:
        chain: 已初始化的 LLMChain 实例。
        user_input: 用户输入的文本。

    Returns:
        str: AI 的回复文本。异常时返回友好错误信息。
    """
    # 空白输入保护
    if not user_input.strip():
        return "（您发送了空消息，请输入一些内容。）"

    try:
        logger.debug("发送消息到 DeepSeek: %s", user_input[:100])
        response: dict[str, Any] = await chain.ainvoke({"input": user_input})
        text: str = response.get("text", "")
        if not text:
            logger.warning("DeepSeek 返回了空文本")
            return "（模型未生成回复，请重试。）"
        return text

    except ConnectionError as exc:
        logger.error("连接 DeepSeek API 失败: %s", exc)
        return (
            "⚠️ 无法连接到 DeepSeek API。请检查：\n"
            "  1. 网络是否正常\n"
            "  2. DEEPSEEK_API_KEY 是否正确\n"
            "  3. DEEPSEEK_BASE_URL 是否可达"
        )

    except TimeoutError:
        logger.error("DeepSeek API 请求超时")
        return "⚠️ 请求超时。DeepSeek 服务响应较慢，请稍后重试。"

    except Exception as exc:
        # 捕获 openai 包的所有异常（RateLimitError、APIStatusError 等）
        error_message = str(exc)
        logger.error("DeepSeek API 调用异常: %s", error_message)

        # 根据错误信息给出更具体的提示
        if "rate" in error_message.lower() or "429" in error_message:
            return "⚠️ API 调用频率过高，请稍等片刻再试。"
        if "auth" in error_message.lower() or "401" in error_message or "403" in error_message:
            return "⚠️ API 密钥验证失败。请检查 DEEPSEEK_API_KEY 是否有效。"
