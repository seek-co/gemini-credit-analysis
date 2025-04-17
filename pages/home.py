import dash
import dash_bootstrap_components as dbc
from dash import html

from src.project_cards import featured_project_cards, create_project_card


dash.register_page(__name__, title="AI Credit Analysis Project", path='/', order=0)


def layout():
    return html.Div(
    children=[
        dbc.Row(
            [
                html.Br(),
                html.H5("AI Agents Entry Point", className="mt-5 mb-2", style={"text-align": "center"})
            ],
        ),
        # project cards
        dbc.Row(
            [
                dbc.Col(
                    create_project_card(
                        img_src=f"/assets/{card['image_file']}",
                        title=f"{card['title']}",
                        description=card["description"],
                        button_href=f"/projects/{page_name}"
                    ), width="auto"
                ) for page_name, card in featured_project_cards.items()
            ], justify="center"
        ),
    ],
)
