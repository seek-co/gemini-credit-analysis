import os
import json
import dash
import dash_bootstrap_components as dbc

from dash import html, dcc, callback, Input, Output, State

from src.news_alert import evaluate_credit_impact, gmail_authenticate, build_message


# load .env if not in production
if not bool(os.getenv("PRODUCTION_MODE")):
    from dotenv import load_dotenv
    load_dotenv()


dash.register_page(__name__, path="/projects/alert", name="Opportunity Alert", order=1)


with open('./assets/news/news_sasol.json', "r") as f:
    news_sasol_ls = json.loads(f.read())
with open('./assets/news/news_cenomi.json', "r") as f:
    news_cenomi_ls = json.loads(f.read())
with open('./assets/news/news_tullow.json', "r") as f:
    news_tullow_ls = json.loads(f.read())


news_sasol_buttons = [
    dbc.Button(
        dbc.CardBody(
            [
                html.H6(children=news['title'], id=f"news_sasol_btn_title_{i}"),
                html.Div(children=news['body'][:100] + " ...", id=f"news_sasol_btn_body_trunc{i}"),
                html.Div(children=news['body'], id=f"news_sasol_btn_body_{i}", hidden=True),
            ]
        ),
        n_clicks=0,
        style={"text-align": "left"},
        id=f"news_sasol_btn_{i}"
    ) for i, news in enumerate(news_sasol_ls)
]
news_cenomi_buttons = [
    dbc.Button(
        dbc.CardBody(
            [
                html.H6(children=news['title'], id=f"news_cenomi_btn_title_{i}"),
                html.Div(children=news['body'][:100] + " ...", id=f"news_cenomi_btn_body_trunc_{i}"),
                html.Div(children=news['body'], id=f"news_cenomi_btn_body_{i}", hidden=True),
            ]
        ),
        n_clicks=0,
        style={"text-align": "left"},
        id=f"news_cenomi_btn_{i}"
    ) for i, news in enumerate(news_cenomi_ls)
]
news_tullow_buttons = [
    dbc.Button(
        dbc.CardBody(
            [
                html.H6(children=news['title'], id=f"news_tullow_btn_title_{i}"),
                html.Div(children=news['body'][:100] + " ...", id=f"news_tullow_btn_body_trunc_{i}"),
                html.Div(children=news['body'], id=f"news_tullow_btn_body_{i}", hidden=True),
            ]
        ),
        n_clicks=0,
        style={"text-align": "left"},
        id=f"news_tullow_btn_{i}"
    ) for i, news in enumerate(news_tullow_ls)
]


def layout():
    return dbc.Container(
        children=[
            dbc.Row(
                    dbc.Label("Email Address To Receive Alerts")

            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Input(
                        id="recipient_email_add", placeholder="Enter email address to receive alerts", type="email"
                    )),
                    dbc.Col(dbc.Button("Enter Email", id="recipient_email_btn", n_clicks=0)),
                ]
            ),
            dbc.Row(html.Br()),
            dbc.Row(dbc.Checkbox(id="sasol_use_true_report", label="For Sasol: Use 'true' report 26 Feb", value=False)),
            dbc.Row(html.Div(id="email_address_display")),
            dbc.Row(html.Br()),
            dbc.Row(html.Div(id="alert_result_display")),
            dbc.Row(dbc.Spinner(
                children=html.Div(id="alert_email_loading"),
                type="border", delay_show=10, delay_hide=10, fullscreen=True,
                fullscreen_style={'backgroundColor': 'transparent'}
            )),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            dbc.Row(html.H5("Sasol News")),
                            dbc.Row(news_sasol_buttons)
                        ],
                        width=3
                    ),
                    dbc.Col(width=1),
                    dbc.Col(
                        children=[
                            dbc.Row(html.H5("Cenomi News")),
                            dbc.Row(news_cenomi_buttons)
                        ],
                        width=3
                    ),
                    dbc.Col(width=1),
                    dbc.Col(
                        children=[
                            dbc.Row(html.H5("Tullow News")),
                            dbc.Row(news_tullow_buttons)
                        ],
                        width=3
                    ),

                ],
            )
        ],
        fluid=False
    )


@callback(
    [Output('recipient_email_btn', 'n_clicks'),
     Output('email_address_display', 'children')],
    [Input('recipient_email_btn', 'n_clicks'),
     Input('recipient_email_add', 'value')],
    prevent_initial_call=True
)
def update_recipient_email_address(num_click, email_address):
    if num_click > 0:
        if email_address is not None:
            return 0, f"Alert emails will be sent to: {email_address}"
        else:
            return 0, dash.no_update
    else:
        raise dash.exceptions.PreventUpdate


for i, news in enumerate(news_sasol_ls):
    @callback(
        [Output(f'news_sasol_btn_{i}', 'n_clicks'),
         Output("alert_email_loading", "children", allow_duplicate=True),
         Output('alert_result_display', 'children', allow_duplicate=True)],
        [Input(f'news_sasol_btn_{i}', 'n_clicks'),
         Input("sasol_use_true_report", "value"),
         Input("recipient_email_add", "value"),
         Input(f'news_sasol_btn_title_{i}', 'children'),
         Input(f'news_sasol_btn_body_{i}', 'children')],
        prevent_initial_call=True
    )
    def evaluate_sasol_income_news(num_click, sasol_true_rpt, recipient_email_add, news_title, news_body):
        if num_click > 0:
            if recipient_email_add is not None and news_title is not None and news_body is not None:
                company_name = "Sasol Limited"
                response_json = evaluate_credit_impact(
                    company_name=company_name, news_title=news_title, news_body=news_body,
                    sasol_true_rpt=sasol_true_rpt
                )
                if isinstance(response_json, dict):
                    service = gmail_authenticate()
                    # construct email
                    if response_json['credit_impact'] != "neutral":
                        email_subject = f"Possible {response_json['credit_impact']} opportunity for {company_name}"
                        email_body = (
                            f"Opportunity alert: {response_json['credit_impact']} \n\n"
                            f"Due to news: {response_json['news_title']} \n\n"
                            f"News summary: {response_json['news_summary']} \n\n"
                            f"Basis: {response_json['basis']} \n\n"
                            f"Reason: {response_json['reason']}"
                        )
                        # send email
                        service.users().messages().send(
                              userId="me",
                              body=build_message(
                                  from_email=os.getenv("GMAIL_SENDER_EMAIL"), destination=recipient_email_add,
                                  subject=email_subject, body=email_body
                              )
                            ).execute()
                    return 0, None, [html.Div(f"{key}: {value}") for key, value in response_json.items()]
                else:
                    return 0, None, f"{response_json}"
            else:
                return 0, dash.no_update, dash.no_update
        else:
            raise dash.exceptions.PreventUpdate


for i, news in enumerate(news_cenomi_ls):
    @callback(
        [Output(f'news_cenomi_btn_{i}', 'n_clicks'),
         Output("alert_email_loading", "children", allow_duplicate=True),
         Output('alert_result_display', 'children', allow_duplicate=True)],
        [Input(f'news_cenomi_btn_{i}', 'n_clicks'),
         Input("recipient_email_add", "value"),
         Input(f'news_cenomi_btn_title_{i}', 'children'),
         Input(f'news_cenomi_btn_body_{i}', 'children')],
        prevent_initial_call=True
    )
    def evaluate_cenomi_income_news(num_click, recipient_email_add, news_title, news_body):
        if num_click > 0:
            if recipient_email_add is not None and news_title is not None and news_body is not None:
                company_name = "Arabian Centres"
                response_json = evaluate_credit_impact(
                    company_name=company_name, news_title=news_title, news_body=news_body, sasol_true_rpt=False
                )
                if isinstance(response_json, dict):
                    service = gmail_authenticate()
                    # construct email
                    if response_json['credit_impact'] != "neutral":
                        email_subject = f"Possible {response_json['credit_impact']} opportunity for {company_name}"
                        email_body = (
                            f"Opportunity alert: {response_json['credit_impact']} \n\n"
                            f"Due to news: {response_json['news_title']} \n\n"
                            f"News summary: {response_json['news_summary']} \n\n"
                            f"Basis: {response_json['basis']} \n\n"
                            f"Reason: {response_json['reason']}"
                        )
                        # send email
                        service.users().messages().send(
                              userId="me",
                              body=build_message(
                                  from_email=os.getenv("GMAIL_SENDER_EMAIL"), destination=recipient_email_add,
                                  subject=email_subject, body=email_body
                              )
                            ).execute()
                    return 0, None, [html.Div(f"{key}: {value}") for key, value in response_json.items()]
                else:
                    return 0, None, f"{response_json}"
            else:
                return 0, dash.no_update, dash.no_update
        else:
            raise dash.exceptions.PreventUpdate


for i, news in enumerate(news_tullow_ls):
    @callback(
        [Output(f'news_tullow_btn_{i}', 'n_clicks'),
         Output("alert_email_loading", "children", allow_duplicate=True),
         Output('alert_result_display', 'children', allow_duplicate=True)],
        [Input(f'news_tullow_btn_{i}', 'n_clicks'),
         Input("recipient_email_add", "value"),
         Input(f'news_tullow_btn_title_{i}', 'children'),
         Input(f'news_tullow_btn_body_{i}', 'children')],
        prevent_initial_call=True
    )
    def evaluate_tullow_income_news(num_click, recipient_email_add, news_title, news_body):
        if num_click > 0:
            if recipient_email_add is not None and news_title is not None and news_body is not None:
                company_name = "Tullow Oil"
                response_json = evaluate_credit_impact(
                    company_name=company_name, news_title=news_title, news_body=news_body, sasol_true_rpt=False
                )
                if isinstance(response_json, dict):
                    service = gmail_authenticate()
                    # construct email if bearish or bullish
                    if response_json['credit_impact'] != "neutral":
                        email_subject = f"Possible {response_json['credit_impact']} opportunity for {company_name}"
                        email_body = (
                            f"Opportunity alert: {response_json['credit_impact']} \n\n"
                            f"Due to news: {response_json['news_title']} \n\n"
                            f"News summary: {response_json['news_summary']} \n\n"
                            f"Basis: {response_json['basis']} \n\n"
                            f"Reason: {response_json['reason']}"
                        )
                        # send email
                        service.users().messages().send(
                              userId="me",
                              body=build_message(
                                  from_email=os.getenv("GMAIL_SENDER_EMAIL"), destination=recipient_email_add,
                                  subject=email_subject, body=email_body
                              )
                            ).execute()
                    return 0, None, [html.Div(f"{key}: {value}") for key, value in response_json.items()]
                else:
                    return 0, None, f"{response_json}"
            else:
                return 0, dash.no_update, dash.no_update
        else:
            raise dash.exceptions.PreventUpdate

