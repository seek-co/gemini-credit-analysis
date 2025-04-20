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
            
            # New components for manual news input and file upload
            dbc.Row(dbc.Label("Enter News Manually")),
            dbc.Row(
                dbc.Textarea(
                    id="manual_news_input",
                    placeholder="Enter raw news content - we'll analyze it for you...",
                    style={"height": "150px"}
                )
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Select Company:"), width=2),
                    dbc.Col(
                        dcc.Dropdown(
                            id="manual_news_company",
                            options=[
                                {"label": "Sasol Limited", "value": "Sasol Limited"},
                                {"label": "Arabian Centres", "value": "Arabian Centres"},
                                {"label": "Tullow Oil", "value": "Tullow Oil"}
                            ],
                            placeholder="Select company (optional - will auto-detect if blank)"
                        ),
                        width=4
                    ),
                    dbc.Col(
                        dbc.Button("Process News", id="process_manual_news_btn", n_clicks=0),
                        width=4
                    ),
                    dbc.Col(
                        html.Small("AI will detect the company, title, and content automatically.", className="text-muted"),
                        width=12,
                        className="mt-1"
                    )
                ],
                className="mt-2"
            ),
            dbc.Row(html.Br()),
            
            dbc.Row(dbc.Label("Or Upload News File")),
            dbc.Row(
                dcc.Upload(
                    id="news_upload",
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    },
                    accept=".json, .txt, .pdf, .md",
                    multiple=False
                )
            ),
            dbc.Row(
                dbc.Button("Process Uploaded File", id="process_upload_btn", n_clicks=0, className="mt-2"),
            ),
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
                print(response_json)
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


# Callback to handle uploaded news file
@callback(
    Output("manual_news_input", "value"),
    Input("news_upload", "contents"),
    State("news_upload", "filename"),
    prevent_initial_call=True
)
def update_news_from_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate
    
    content_type, content_string = contents.split(',')
    
    import base64
    decoded = base64.b64decode(content_string)
    
    try:
        if filename.endswith('.json'):
            # Parse JSON file
            import json
            data = json.loads(decoded.decode('utf-8'))
            if isinstance(data, dict):
                # Join all values in the dictionary
                return ' '.join(str(v) for v in data.values())
            elif isinstance(data, list) and len(data) > 0:
                # Take the first item if it's a list
                if isinstance(data[0], dict):
                    return ' '.join(str(v) for v in data[0].values())
                else:
                    return ' '.join(str(v) for v in data)
            else:
                return decoded.decode('utf-8')
        elif filename.endswith('.txt'):
            # Plain text file
            return decoded.decode('utf-8')
        elif filename.endswith('.md'):
            # Parse Markdown file - just get raw text
            return decoded.decode('utf-8')
        elif filename.endswith('.pdf'):
            # Parse PDF file
            import io
            try:
                import pdfplumber
                pdf_file = io.BytesIO(decoded)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                return "PDF parsing library (pdfplumber) not installed. Please install with 'pip install pdfplumber'."
        else:
            return f"Unsupported file type: {filename}\n\nPlease upload a .json, .txt, .pdf, or .md file."
    except Exception as e:
        return f"Error processing file: {str(e)}"


# Callback to process manually entered news
@callback(
    [Output('process_manual_news_btn', 'n_clicks'),
     Output("alert_email_loading", "children", allow_duplicate=True),
     Output('alert_result_display', 'children', allow_duplicate=True)],
    [Input('process_manual_news_btn', 'n_clicks'),
     Input("recipient_email_add", "value"),
     Input("manual_news_input", "value"),
     Input("manual_news_company", "value"),
     Input("sasol_use_true_report", "value")],
    prevent_initial_call=True
)
def process_manual_news(num_click, recipient_email_add, news_text, selected_company, sasol_true_rpt):
    if num_click > 0:
        if recipient_email_add is None or news_text is None:
            return 0, None, "Please provide email address and news text"
        
        # Use Gemini to analyze the raw news text
        from src.news_alert import analyze_raw_news
        
        # Get the analysis result
        analysis_result = analyze_raw_news(news_text)
        news_title = analysis_result["title"]
        news_body = analysis_result["body"]

        print(analysis_result)
        
        # Use selected company if provided, otherwise use detected company
        if selected_company:
            company = selected_company
            company_detection_msg = f"Using selected company: {selected_company}"
        else:
            company = analysis_result["company"]
            company_detection_msg = f"Detected company: {company} (confidence: {analysis_result['confidence']})"
        
        # Process with the appropriate company
        use_sasol_true_rpt = sasol_true_rpt if company == "Sasol Limited" else False
        
        response_json = evaluate_credit_impact(
            company_name=company, news_title=news_title, news_body=news_body, 
            sasol_true_rpt=use_sasol_true_rpt
        )

        print("result atfter evaluation: ", response_json)
        
        if isinstance(response_json, dict):
            service = gmail_authenticate()
            # Send email if credit impact is not neutral
            if response_json['credit_impact'] != "neutral":
                email_subject = f"Possible {response_json['credit_impact']} opportunity for {company}"
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
                        from_email=os.getenv("GMAIL_SENDER_EMAIL"), 
                        destination=recipient_email_add,
                        subject=email_subject, 
                        body=email_body
                    )
                ).execute()
            
            # Display analysis info followed by evaluation results
            result_display = [
                html.Div("News Analysis Results:", style={"fontWeight": "bold", "marginTop": "10px"}),
                html.Div(f"Extracted Title: {news_title}"),
                html.Div(company_detection_msg),
                html.Div("Credit Evaluation Results:", style={"fontWeight": "bold", "marginTop": "10px"})
            ]
            result_display.extend([html.Div(f"{key}: {value}") for key, value in response_json.items()])
            return 0, None, result_display
        else:
            return 0, None, f"{company_detection_msg}\n{response_json}"
    else:
        raise dash.exceptions.PreventUpdate


# Callback to process uploaded file
@callback(
    [Output('process_upload_btn', 'n_clicks'),
     Output("alert_email_loading", "children", allow_duplicate=True),
     Output('alert_result_display', 'children', allow_duplicate=True)],
    [Input('process_upload_btn', 'n_clicks'),
     Input("recipient_email_add", "value"),
     Input("manual_news_input", "value"),
     Input("sasol_use_true_report", "value")],
    prevent_initial_call=True
)
def process_uploaded_file(num_click, recipient_email_add, news_text, sasol_true_rpt):
    if num_click > 0:
        if recipient_email_add is None or news_text is None:
            return 0, None, "Please provide email address and upload a file"
        
        # Use Gemini to analyze the raw news text
        from src.news_alert import analyze_raw_news
        
        # Get the analysis result
        analysis_result = analyze_raw_news(news_text)
        news_title = analysis_result["title"]
        news_body = analysis_result["body"]
        company = analysis_result["company"]
        company_detection_msg = f"Detected company: {company} (confidence: {analysis_result['confidence']})"
        
        # Process with the appropriate company
        use_sasol_true_rpt = sasol_true_rpt if company == "Sasol Limited" else False
        
        response_json = evaluate_credit_impact(
            company_name=company, news_title=news_title, news_body=news_body, 
            sasol_true_rpt=use_sasol_true_rpt
        )
        
        if isinstance(response_json, dict):
            service = gmail_authenticate()
            # Send email if credit impact is not neutral
            if response_json['credit_impact'] != "neutral":
                email_subject = f"Possible {response_json['credit_impact']} opportunity for {company}"
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
                        from_email=os.getenv("GMAIL_SENDER_EMAIL"), 
                        destination=recipient_email_add,
                        subject=email_subject, 
                        body=email_body
                    )
                ).execute()
            
            # Display analysis info followed by evaluation results
            result_display = [
                html.Div("News Analysis Results:", style={"fontWeight": "bold", "marginTop": "10px"}),
                html.Div(f"Extracted Title: {news_title}"),
                html.Div(company_detection_msg),
                html.Div("Credit Evaluation Results:", style={"fontWeight": "bold", "marginTop": "10px"})
            ]
            result_display.extend([html.Div(f"{key}: {value}") for key, value in response_json.items()])
            return 0, None, result_display
        else:
            return 0, None, f"{company_detection_msg}\n{response_json}"
    else:
        raise dash.exceptions.PreventUpdate

