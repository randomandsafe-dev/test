"""DeepSeek Chat-Bot 主入口

提供交互式命令行界面，支持多轮对话、记忆重置和优雅退出。

用法示例::

    # 直接运行
    python chatbot.py

    # 或以模块方式运行
    python -m chatbot
"""
from __future__ import annotations

import asyncio
import logging
import sys

from chat_chain import create_chat_chain, run_chain
from config import get_settings
from llm.deepseek import get_deepseek_llm
from memory import clear_memory, get_memory

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # 生产环境用 WARNING，调试时改为 DEBUG
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("chatbot")

# ---------------------------------------------------------------------------
# 交互式命令常量
# ---------------------------------------------------------------------------
CMD_RESET = ":reset"
CMD_QUIT = ":quit"
CMD_HELP = ":help"
CMD_EXIT = ":exit"

HELP_TEXT = f"""
╔══════════════════════════════════════════════╗
║         DeepSeek Chat-Bot 帮助              ║
╠══════════════════════════════════════════════╣
║  直接输入文字  →  与 DeepSeek 对话          ║
║  {CMD_RESET:<9}  →  重置对话（清空记忆）     ║
║  {CMD_QUIT:<9}  →  退出程序                 ║
║  {CMD_EXIT:<9}  →  退出程序（同 :quit）     ║
║  {CMD_HELP:<9}  →  显示本帮助               ║
╚══════════════════════════════════════════════╝
"""

BANNER = r"""
╔══════════════════════════════════════════════╗
║     🤖 DeepSeek Chat-Bot (LangChain)       ║
║  基于 LangChain + DeepSeek API 构建         ║
║  输入 :help 查看命令 | :quit 退出          ║
╚══════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------
async def chat_loop() -> None:
    """交互式对话主循环。

    初始化 LLM → memory → chain，然后在 while 循环中处理用户输入。
    """
    # ---- 初始化 ----
    try:
        settings = get_settings()
        logger.info("配置加载成功: model=%s", settings.deepseek_model)
    except ValueError as exc:
        print(f"❌ 配置错误: {exc}", file=sys.stderr)
        print("请设置 DEEPSEEK_API_KEY 环境变量后重试。", file=sys.stderr)
        sys.exit(1)

    try:
        llm = get_deepseek_llm()
        memory = get_memory()
        chain = create_chat_chain(llm, memory)
        logger.info("对话链初始化完成")
    except Exception as exc:
        print(f"❌ 初始化失败: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- 打印欢迎信息 ----
    print(BANNER)

    # ---- 主循环 ----
    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        # 处理空输入
        if not user_input:
            continue

        # 处理命令
        if user_input == CMD_QUIT or user_input == CMD_EXIT:
            print("👋 再见！")
            break

        if user_input == CMD_HELP:
            print(HELP_TEXT)
            continue

        if user_input == CMD_RESET:
            memory = clear_memory()
            chain = create_chat_chain(llm, memory)
            print("🔄 对话已重置，记忆已清空。")
            continue

        # ---- 调用 DeepSeek ----
        print("🤖 AI: ", end="", flush=True)
        reply = await run_chain(chain, user_input)
        print(reply)
