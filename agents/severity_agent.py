"""
Severity Agent — evaluates the clinical severity of an adverse event.
Exposed as a LangGraph node function: severity_node(state) -> state update.
"""
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from .tools import calculate_severity_score, lookup_drug_class

SYSTEM_PROMPT = """You are a pharmacovigilance specialist focused on adverse event severity assessment.

Given an adverse event report, your job is to:
1. Use the calculate_severity_score tool to score the severity of the event
2. Use the lookup_drug_class tool to identify the drug's pharmacological class
3. Return a concise severity assessment (3-5 sentences) including the score and drug class

Be factual and concise. Do not invent information not present in the report."""

_llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)
_agent = create_react_agent(_llm, tools=[calculate_severity_score, lookup_drug_class])


def severity_node(state: dict) -> dict:
    """LangGraph node: runs the severity agent on the AE report."""
    result = _agent.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Assess the severity of this adverse event report:\n\n{state['report']}"},
        ]
    })
    return {"severity_analysis": result["messages"][-1].content}
