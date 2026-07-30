"""DeepSeek Chat-Bot 配置模块

通过环境变量管理所有可配置参数，提供合理的默认值。
支持 .env 文件加载（需 python-dotenv）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """DeepSeek 对话配置

    所有字段均可通过同名环境变量覆盖（全大写形式）。
    优先级：环境变量 > 默认值。
    """

    # DeepSeek API 密钥（必填，无默认值以强制用户配置）
    deepseek_api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "")
    )

    # 模型名称 — DeepSeek 目前推荐的聊天模型
    deepseek_model: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    )

    # API 基础地址（OpenAI 兼容接口）
    deepseek_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
    )

    # 生成温度（0-2，越低越确定，越高越随机）
    deepseek_temperature: float = field(
        default_factory=lambda: float(
            os.environ.get("DEEPSEEK_TEMPERATURE", "0.7")
        )
    )

    # 最大回复 token 数
    deepseek_max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get("DEEPSEEK_MAX_TOKENS", "2048")
        )
    )

    # 请求超时（秒）
    deepseek_timeout: int = field(
        default_factory=lambda: int(
            os.environ.get("DEEPSEEK_TIMEOUT", "60")
        )
    )

    def validate(self) -> None:
        """校验必要配置是否存在。"""
        if not self.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。"
                "请通过环境变量或 .env 文件提供有效的 API 密钥。"
            )


