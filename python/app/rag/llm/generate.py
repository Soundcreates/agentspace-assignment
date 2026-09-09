from __future__ import annotations

import os

ABSTAIN = "I couldn't generate an answer."

SYSTEM_PROMPT = """You are a helpful technical Q&A assistant.

Your own knowledge is the primary source for answering. Retrieved document passages are optional helper context — an addon, not the main constraint.

Rules:
- Keep the core answer short: a few concise sentences (or a short bullet list) from your own knowledge.
- If helper passages are present and relevant, you may add a brief extra section that pulls in useful details from them. Do not dump large passages verbatim.
- If the passages are incomplete, off-topic, or missing, still give the short answer from your own knowledge. Do not refuse just because retrieval was weak.
- Prefer passage details for project-specific names, claims, and facts when they clearly apply.
- Prefer brevity over completeness. Avoid long essays unless the user explicitly asks for depth.
"""


def _message_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def generate_answer(
    question: str,
    context: str,
    llm=None,
    on_token=None,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if llm is None and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter API key before asking questions."
        )

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if llm is None:
        from langchain_openai import ChatOpenAI

        headers: dict[str, str] = {}
        title = os.getenv("OPENROUTER_TITLE", "agentspace-rag")
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if title:
            headers["X-Title"] = title
        if referer:
            headers["HTTP-Referer"] = referer
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0,
            streaming=True,
            default_headers=headers or None,
        )

    from langchain_core.messages import HumanMessage, SystemMessage

    helper = context.strip() if context.strip() else "(no helper passages retrieved)"
    user_prompt = (
        "OPTIONAL HELPER PASSAGES from the user's files (addon context, not the only source):\n"
        f"{helper}\n\n"
        f"QUESTION: {question}\n\n"
        "Give a short answer from your own knowledge first. "
        "Only if the helper passages clearly add useful project-specific detail, append a brief "
        "context-backed note — keep that short too, even if the passages are long."
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    if on_token is not None:
        parts: list[str] = []
        for chunk in llm.stream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else _message_text(chunk.content)
            if piece:
                on_token(piece)
                parts.append(piece)
        text = "".join(parts).strip()
    else:
        response = llm.invoke(messages)
        text = (
            response.content if isinstance(response.content, str) else _message_text(response.content)
        ).strip()
    return text or ABSTAIN
