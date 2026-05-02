"""
Summary Agent — synthesizes findings from all specialist agents into a final report.
Exposed as a LangGraph node function: summary_node(state) -> state update.
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a senior medical writer producing structured safety reports.

You will receive:
- The original adverse event report
- A severity assessment from the Severity Agent
- A compliance assessment from the Compliance Agent

Synthesize these into a structured report with three sections:
1. **Severity Summary** — key severity findings and score
2. **Compliance Status** — whether the report is complete, any gaps
3. **Recommended Action** — one clear next step (e.g. "Submit as is", "Obtain missing onset date before submission")

Keep the report under 200 words. Be direct and actionable."""
_llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)


def summary_node(state: dict) -> dict:
    """LangGraph node: synthesizes all findings into a final structured report."""
    prompt = f"""Original Report:
{state['report']}

Severity Assessment:
{state['severity_analysis']}

Compliance Assessment:
{state['compliance_analysis']}

Please produce the structured safety report."""

    response = _llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {"final_report": response.content}
