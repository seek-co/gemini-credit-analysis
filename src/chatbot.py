import markdown
import dash_bootstrap_components as dbc
from dash import html, dcc

from src.util import gpt_o1_loop_reasoning_with_tools, ReasoningToolSchema


def render_chat_input():
    chat_input = dbc.InputGroup(
        children=[
            dbc.Textarea(
                id="chatbot_user_input", placeholder="Ask a question about the selected company...",
                rows=3
            ),
            dbc.Button(id="chatbot_send_button", children=">", n_clicks=0),
        ],
    )
    return chat_input

def render_message(text:str, box:str = "AI"):
    text = text.replace(f"ChatBot:", "").replace("Human:", "")
    style = {
        "font-size": "14px",
        "max-width": "80%",
        "width": "max-content",
        "padding": "2px 4px",
        "border-radius": 10,
        # "margin-top": 5,
        "margin-bottom": 6,
        'border': '0px solid'
    }
    if box == "human":
        style["margin-left"] = "auto"
        style["margin-right"] = 0
        style["color"] = "black"
        # markdown.markdown(html.escape(message), extensions=['nl2br'])
        textbox_human = dbc.Card(
            dcc.Markdown(
                markdown.markdown(text, extensions=['nl2br', 'markdown.extensions.tables']),
                dangerously_allow_html=True
            ),
            style=style, body=True, color='light', inverse=False
        )
        return html.Div([textbox_human])
    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"
        style["max-width"] = "95%"
        style["color"] = "white"

        textbox = dbc.Card(
            dcc.Markdown(
                markdown.markdown(text, extensions=['nl2br', 'markdown.extensions.tables']),
                dangerously_allow_html=True
            ),
            style=style, body=True, color="#0d6efd", inverse=True
        )
        return html.Div([textbox])
    else:
        raise ValueError("Incorrect option for `box`.")


class ChatBot:
    def __init__(self, llm: str, reasoning_effort: str = "high"):
        self.llm = llm
        self.reasoning_effort = reasoning_effort
        self.chat_history = []

    def generate_response(self, company_name, user_prompt):
        tl = ReasoningToolSchema()
        tools = [
            tl.rag_retrieval_tool, tl.gpt_4o_web_search_tool, tl.get_financial_metrics_yf_api_tool,
            tl.get_bonds_api_for_chatbot_tool
        ]
        file_categories = ["financial", "bond", "ratings", "additional", "generated"]
        self.chat_history.append({
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}]
        })
        response_str, step_outputs = gpt_o1_loop_reasoning_with_tools(
            model_name=self.llm, reasoning_effort=self.reasoning_effort, input_messages=self.chat_history,
            tools=tools, file_category=file_categories, company_name=company_name #
        )
        self.chat_history.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": response_str}]
        })
        return response_str, step_outputs
