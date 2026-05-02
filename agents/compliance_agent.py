"""
Compliance Agent — checks whether the AE report contains all required fields.
Exposed as a LangGraph node function: compliance_node(state) -> state update.
"""
import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from .tools import check_required_fields

SYSTEM_PROMPT = """You are a regulatory affairs specialist focused on adverse event report completeness.

Given an adverse event report, your job is to:
1. Use the check_required_fields tool to identify any missing ICH E2B required fields
2. Return a concise compliance assessment stating which fields are present, which are missing,
   and whether the report is suitable for regulatory submission

Be factual and concise. Do not invent information not present in the report."""

_llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)
_agent = create_react_agent(_llm, tools=[check_required_fields])


def compliance_node(state: dict) -> dict:
    """LangGraph node: runs the compliance agent on the AE report."""
    result = _agent.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Check the completeness of this adverse event report:\n\n{state['report']}"},
        ]
    })
    return {"compliance_analysis": result["messages"][-1].content}
