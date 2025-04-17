import dash_bootstrap_components as dbc

from dash import dcc, html
from collections import OrderedDict


template = {
    "title": "Card title",
    "description": '''Some quick example text to build on the card title and make up the bulk of the card's content.''',
    "image_file": "placeholder286x180.png",
    "page_name": "projectname"
}

placeholder_img = "placeholder286x180.png"

ai_project_cards = OrderedDict([
    ('dashboard', {
        'title': 'AI Credit Report & Chatbot',
        'description': "AI credit report generation and interactive chatbot for users query on a company's financials.",
        "image_file": "report.jpeg",
    }),
    # ('chatbot', {
    #     'title': 'Q&A Chatbot',
    #     'description': "An interactive chatbot that allows users to query about a company's credit report.",
    #     "image_file": "chatbot.jpeg",
    # }),
    ('alert', {
        'title': 'Opportunity Alert',
        'description': "An AI alert system notifying users on possible trading opportunities resulted from news events.",
        "image_file": "alert.jpeg",
    }),
])

featured_project_cards = OrderedDict(
    [(pgname, content) for pgname, content in ai_project_cards.items() if pgname in ['dashboard', 'alert']]
)

def create_project_card(img_src, title, description, button_href):
    return dbc.Card(
        [
            # link embed in image
            dbc.CardLink(
                [
                    dbc.CardImg(
                        src=img_src,
                        top=True,
                        style={
                            'border-top-left-radius': '8px',
                            'border-top-right-radius': '8px'
                        }
                    )
                ],
                href=button_href
            ),
            dbc.CardBody(
                [
                    # card title
                    html.H4(
                        dcc.Link(
                            title,
                            href=button_href,
                            style={
                                'font-size': '20px',
                                "color": "inherit",
                                # "hover-color": '#593196',
                                "text-decoration": "none"
                            }
                        ),
                        className="card-title"
                    ),
                    # card description text
                    html.P(description, className="card-text", style={'font-size': '15px'}),
                    # view more button
                    html.Div(
                        dbc.Button(
                            ["View", html.I(className="bi bi-arrow-right ms-1")],
                            href=button_href,
                            outline=True,
                            style={'font-size': '15px', "text-align": "right", 'color': '#64a8fa', 'border': '0px solid'}
                        ),
                        className="d-grid gap-2 col-12 mx-auto"
                    )
                ]
            ),
        ],
        style={"width": "18rem", "margin": "10px", 'border-radius': '8px'},
    )