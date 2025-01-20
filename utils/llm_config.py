import asyncio
from openai import AsyncOpenAI


AMAP_API_KEY = "CiAukOSFzSmHuPgZq6w5QpdC"
# 线上环境
AMAP_BASE_URL = "https://llm-proxy.alibaba-inc.com/v1"
AMAP_CACHE_URL = "https://llm-cache-proxy.amap.com/v1"

AI_STUDIO_API_KEY = "3dc91ae53e03073848e8427d90346a53"
# AI_STUDIO_BASE_URL = "https://idealab.alibaba-inc.com/api/openai/v1"
AI_STUDIO_BASE_URL = "https://llm-proxy.alibaba-inc.com/api/openai/v1"


# 个人key
# PRIVATE_GEMINI_API_KEY = "AIzaSyAvfhdbKel0nsOUpIiCbG48dFiyAjEdreE"
PRIVATE_GEMINI_API_KEY = "AIzaSyA1IkPq790mondjYph8z6g3HjDji_R5G-Q"
PRIVATE_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"

client = AsyncOpenAI(
    base_url=AMAP_BASE_URL,
    api_key="CiAukOSFzSmHuPgZq6w5QpdC",
)


def ceate_chat_client(
    model,
    api_key,
    base_url,
    stream=True,
    temperature=0,
    n=1,
    logprobs=False,
    max_tokens=4096,
    model_alias_name="",
    **args
):
    """
    使用示例
        stream_data = await async_llm_chat([
            {"role": "user", "content": '帮我实现一个微信聊天功能, 给我具体的代码'},
        ])
        返回的数据格式通过以下方式进行读取
        async for chunk in stream_data:
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    async def invoke(messages=[], **inner_args):
        if not messages:
            raise Exception("入参不能为空")
        final_args = {
            "stream": stream,
            "model": model,
            "logprobs": logprobs,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **args,
            **inner_args,
            "messages": messages,
        }
        if model_alias_name == "gemini_private":
            del final_args["logprobs"]
        response = await client.chat.completions.create(**final_args)

        return response

    return invoke


llm_4o = ceate_chat_client(
    base_url=AMAP_BASE_URL,
    api_key=AMAP_API_KEY,
    model="gpt4o",
)
llm_4o_nocache = ceate_chat_client(
    base_url=AMAP_BASE_URL,
    api_key=AMAP_API_KEY,
    model="gpt4o",
)
llm_4o_cache = ceate_chat_client(
    base_url=AMAP_CACHE_URL,
    api_key=AMAP_API_KEY,
    model="gpt4o",
)
llm_claude = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="claude35_sonnet",
    max_tokens=8192,
)
llm_claude_s2 = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="claude35_sonnet2",
    max_tokens=8192,
)
llm_4o_idea = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="gpt-4o-0806-global",
    max_tokens=16384,
)
llm_gemini = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="gemini-1.5-pro",
)
llm_gemini_aimap = ceate_chat_client(
    base_url=AMAP_BASE_URL,
    api_key=AMAP_API_KEY,
    model="gemini-1.5-pro",
)
llm_qianwen = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="qwen2.5-72b-instruct",
)
llm_qianwenmax = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="qwen_max",
)
llm_qianwenPlus = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="qwen-plus",
)
llm_o1 = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="o1-preview-0912",
    stream=False,
)
llm_o1_mini = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="o1-mini-0912",
    stream=False,
)
llm_4o_aistudio = ceate_chat_client(
    base_url=AI_STUDIO_BASE_URL,
    api_key=AI_STUDIO_API_KEY,
    model="gpt-4o-0806",
    # model="gpt-4o-0513",
    # model="gpt-4o-mini-0718",
    max_tokens=16384,
)
llm_gemini_private = ceate_chat_client(
    base_url=PRIVATE_GEMINI_BASE_URL,
    api_key=PRIVATE_GEMINI_API_KEY,
    model="gemini-1.5-pro",
    model_alias_name="gemini_private"
)
