import pathlib
import json
import os
import io
import re
import base64
import dash
import dash_bootstrap_components as dbc
import markdown
import weaviate
import weaviate.classes.query as wq

from weaviate.classes.tenants import Tenant
from weaviate.classes.init import Auth
from dash import html, dcc, callback, Input, Output, State
from google.cloud import storage
from unstructured.partition.md import partition_md

from src.util import (
    gcloud_connect_credentials, get_company_list, extract_data_list, ingest_all_data,
    parse_new_company_file_to_vector_db, create_gcs_folder, delete_gcs_folder, change_company_name_gsc_to_vectordb
)
from src.report_gen import CreditReportGenerate
from src.chatbot import ChatBot, render_message, render_chat_input


# load .env if not in production
if not bool(os.getenv("PRODUCTION_MODE")):
    from dotenv import load_dotenv
    load_dotenv()


dash.register_page(__name__, path="/projects/dashboard", name="AI Credit Report & Chatbot Dashboard", order=0)

# load company list
company_list = get_company_list()
# create the chatbot object that will save the chat history
chatbot = ChatBot(llm="o1", reasoning_effort="high")


def layout():
    return html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Label("Select Company"),
                    dcc.Dropdown(
                        id="company_dropdown",
                        options=company_list,
                        value="Sasol Limited",
                        multi=False,
                        optionHeight=50,
                        style={'color': 'black'},
                    ),
                ],
            ),
            html.Br(),
            # dbc.Row(
            #     [
            #         dbc.Col(dbc.Label("Quantitative Formula"), width='100%'),
            #         dbc.Col(
            #             dcc.Dropdown(
            #                 id="quant_formula_dropdown",
            #                 options=quant_list,
            #                 # value=["AAPL", "NVDA", "MSFT", ],  # "GOOG", "AMZN"
            #                 multi=True,
            #                 optionHeight=50,
            #                 style={'color': 'black'},
            #                 disabled=True
            #             ),
            #         ),
            #     ],
            # ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        dbc.Button(
                            children="Generate Report", id="generate_report_button", n_clicks=0, color="primary",
                            className="mb-3"
                        ),
                        id="generate_report_button_div", className="d-grid gap-2 col-6 mx-auto"
                    ),
                )
            ),
            dbc.Row(dbc.Label("Manage Company Folders Files")),
            dbc.Row(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            dbc.Row([
                                dbc.Col(
                                    [
                                        dbc.Row(dbc.Input(
                                            id="new_company_name", placeholder="New company name...", type="text"
                                        )),
                                        dbc.Row(html.Div(id="add_company_success")),
                                    ],
                                    width=3
                                ),
                                dbc.Col(dbc.Button("Add New", id="add_company_btn", n_clicks=0)),
                                dbc.Col(dbc.Spinner(
                                    children=html.Div(id="add_company_loading"),
                                    type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                    fullscreen_style={'backgroundColor': 'transparent'}
                                )),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            dcc.RadioItems(
                                                id="new_file_category",
                                                options=[
                                                    {"label": "Annual Reports", "value": "financial"},
                                                    {"label": "Sell-Side Ratings", "value": "ratings"},
                                                    {"label": "Bond Docs", "value": "bond"},
                                                    {"label": "Additional", "value": "additional"}
                                                ],
                                                value=None, inline=True, labelStyle={"align-items": "center"},
                                                inputStyle={"margin-left": "20px", "margin-right": "4px"},
                                            )
                                        ),
                                        dbc.Row([
                                            dbc.Col(html.Div(
                                                dcc.Upload(
                                                    id='add_files_upload',
                                                    children=html.Div([html.A('Select Files')]),
                                                    style={
                                                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                                        'borderWidth': '1px', 'borderStyle': 'dashed',
                                                        'borderRadius': '5px', 'textAlign': 'center',
                                                        'margin': '10px'
                                                    },
                                                    # Allow multiple files to be uploaded
                                                    multiple=True
                                                )
                                            )),
                                            dbc.Col(dbc.Button(
                                                "Add File(s)", id="add_files_btn", n_clicks=0,
                                                className="my-3"
                                            ))
                                        ]),
                                        dbc.Row(dbc.Spinner(
                                            children=html.Div(id='add_files_success'),
                                            type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                            fullscreen_style={'backgroundColor': 'transparent'}
                                        ))
                                    ],
                                    width=6
                                ),
                            ]),
                            title="Add new company or file",
                        ),
                        dbc.AccordionItem(
                            dbc.Row([
                                dbc.Col(
                                    children=[
                                        dbc.Button(
                                            children="Confirm Deleting Current Company?", id="remove_company_btn",
                                            color="danger", n_clicks=0
                                        ),
                                        # html.Br(),
                                        html.Div(id="remove_company_success")
                                    ],
                                    width=4
                                ),
                                dbc.Col(dbc.Spinner(
                                    children=html.Div(id="remove_company_loading"),
                                    type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                    fullscreen_style={'backgroundColor': 'transparent'}
                                )),
                                dbc.Col([dcc.Dropdown(
                                    id="files_dropdown",
                                    options=[],
                                    multi=True,
                                    optionHeight=50,
                                    style={'color': 'black'},
                                    placeholder="Select Files To Remove"
                                ), html.Div(id="remove_files_success")], width=4),
                                dbc.Col(
                                    dbc.Button("Remove", id="remove_files_btn", color="danger", n_clicks=0)
                                ),
                                dbc.Col(dbc.Spinner(
                                    children=html.Div(id="remove_files_loading"),
                                    type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                    fullscreen_style={'backgroundColor': 'transparent'}
                                ))
                            ]),
                            title="Remove company or file",
                        ),
                    ],
                )
            ),
            dbc.Row([
                dbc.Col(
                    children=[
                        dbc.Row(
                            [
                                html.H1(
                                    children="AI Credit Report", className="mt-5 mb-2", style={"text-align": "center"}
                                ),
                                html.Div(
                                    dbc.Row([
                                        dbc.Button(
                                            children="Parse Report to Vector Database",
                                            id="ingest_report_to_vector_db_btn",
                                            n_clicks=0,
                                            color="primary",
                                            className="mb-3",
                                            disabled=True
                                        ),
                                        dbc.Spinner(
                                            children=html.Div(id="parse_report_loading_comp"),
                                            type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                            fullscreen_style={'backgroundColor': 'transparent'}
                                        ),
                                        dbc.Button(
                                            children="Download Report",
                                            id="download_report_btn",
                                            n_clicks=0,
                                            color="primary",
                                            className="mb-3",
                                            disabled=True
                                        ),
                                        dbc.Spinner(
                                            children=html.Div(id="download_report_loading_comp"),
                                            type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                            fullscreen_style={'backgroundColor': 'transparent'}
                                        ),
                                        dcc.Download(id="download_report"),
                                        dcc.Download(id="download_report_html"),
                                    ]),
                                    id="ingest_report_to_vector_db_div", className="d-grid gap-2 col-6 mx-auto"
                                )
                            ]
                        ),
                        dbc.Row(
                            children=[
                                html.Div(
                                    dcc.Markdown(
                                        children="", mathjax=True, dangerously_allow_html=True, id="credit_report"
                                    ),
                                    id='credit_report_div'
                                ),
                                dbc.Spinner(
                                    children=html.Div(id="generate_report_loading_comp"),
                                    type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                    fullscreen_style={'backgroundColor': 'transparent'}
                                ),
                            ],
                            className='me-1',
                            style={
                                'maxHeight': 'calc(150vh - 480px)',
                                'overflow-y': 'scroll',
                            }
                        ),
                    ],
                    width=5
                ),
                dbc.Col(
                    children=[
                        dbc.Row(
                            html.H1("Q&A Chatbot", className="mt-5 mb-2", style={"text-align": "center"})
                        ),
                        dbc.Row(
                            html.Div([
                                dbc.Button(
                                    children="Initialise Chatbot", id="init_chatbot_button", n_clicks=0,
                                    color="primary", className="mb-3"
                                ),
                                dbc.Spinner(
                                    children=html.Div(
                                        children=dbc.Alert(
                                            children="Chatbot Not Initialised", color="light",
                                            style={"textAlign": "center"}
                                        ),
                                        id='init_chatbot_success'
                                    ),
                                    type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                    fullscreen_style={'backgroundColor': 'transparent'}
                                )],
                                id="init_chatbot_button_div", className="d-grid gap-2 col-6 mx-auto"
                            )
                        ),
                        dbc.Row(
                            children=[
                                html.Div(
                                    [
                                        dcc.Store(id="conversation_store", data="[]"),
                                        dbc.Container(
                                            fluid=True,
                                            children=dbc.Row(dbc.Card(
                                                dbc.CardBody([
                                                    html.Div(
                                                        html.Div(id="display_conversation"),
                                                        style={
                                                            "overflow-y": "auto",
                                                            "display": "flex",
                                                            "height": "calc(130vh - 480px)",
                                                            "flex-direction": "column-reverse",
                                                        },
                                                    ),
                                                    html.Br(),
                                                    html.Div(
                                                        render_chat_input(),
                                                        style={
                                                            'margin-left': '70px',
                                                            'margin-right': '70px',
                                                            'margin-bottom': '20px'
                                                        }
                                                    ),
                                                    dbc.Spinner(
                                                        html.Div(id="chat_loading_component"),
                                                        type="border", delay_show=10, delay_hide=10, fullscreen=True,
                                                        fullscreen_style={'backgroundColor': 'transparent'}
                                                    ),
                                                ]),
                                                style={
                                                    'border-radius': 25, 'background': '#FFFFFF',
                                                    'border': '0px solid'
                                                }
                                            ))
                                        ),
                                    ],
                                ),
                            ],
                            className='me-1',
                            style={
                                'maxHeight': 'calc(160vh - 480px)',
                                'overflow-y': 'scroll',
                            }
                        ),
                        html.Br(),
                        # dbc.Row(
                        #     [
                        #         dbc.Col(dbc.Label("Quantitative Formula"), width='100%'),
                        #         dbc.Col(
                        #             [dcc.Textarea(
                        #                 id="quant_formula", placeholder="Quantitative Formula", rows=3,
                        #                 className="mb-3 border rounded form-control"
                        #             )],
                        #         ),
                        #     ],
                        # ),
                    ],
                    width=7
                ),
                dbc.Label(id="selected_company_name"),
            ]),
        ],
    )


# add new company folder to gcs and vector db new tenant on Add New click
@callback(
    [Output("add_files_btn", "n_clicks"),
     Output("add_files_success", "children")],  #],
    [Input("add_files_btn", "n_clicks"),
     Input("company_dropdown", "value"),
     Input("new_file_category", "value"),
     Input("add_files_upload", "contents"),
     State("add_files_upload", "filename"),
     State("add_files_upload", "last_modified")],
    prevent_initial_call=True
)
def add_new_files(num_click, company_name, file_category, file_ctn_ls, file_name_ls, date_ls):
    if num_click > 0 and file_ctn_ls is not None:
        if company_name is not None and file_category is not None:
            for ctn, name, date in zip(file_ctn_ls, file_name_ls, date_ls):
                b64_str = re.sub(r"data:.+;base64,", "", ctn)
                # get file type
                file_type_grp = re.search("data:.+;base64,", ctn)
                file_type = file_type_grp.group().replace("data:", "").replace(";base64,", "")
                # turn into string for gcs
                b64_dec = base64.b64decode(b64_str).decode("utf-8")
                # upload to gcs
                storage_client = storage.Client(credentials=gcloud_connect_credentials())
                bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))
                blob = bucket.blob(f'companies/{company_name}/{name}')
                blob.upload_from_string(b64_dec, content_type=file_type)
                storage_client.close()
                # parse content to vector db
                parse_new_company_file_to_vector_db(
                    file_path=os.path.join("./data", name), company_name=company_name, file_category=file_category,
                    file_content=b64_str
                )
                # raw_ctn = html.Div([html.Pre(ctn[0:300]), html.Pre(b64_dec[0:300])])
            return 0, f"File(s) added successfully for company: {company_name}."
        else:
            return 0, "Company or File Category not selected."
    else:
        raise dash.exceptions.PreventUpdate


# remove company folder recursively on click
@callback(
    [Output("remove_files_btn", "n_clicks"),
     Output("remove_files_loading", "children"),
     Output("remove_files_success", "children")],
    [Input("remove_files_btn", "n_clicks"),
     Input("company_dropdown", "value"),
     Input("files_dropdown", "value")],
    prevent_initial_call=True
)
def remove_files_in_gcs_and_vector_db(num_clicks, company_name, company_files):
    if num_clicks > 0:
        if company_name is not None and len(company_files) != 0:
            # remove files in gcs companies folder
            storage_client = storage.Client(credentials=gcloud_connect_credentials())
            bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))
            for file in company_files:
                cmp_blob_name = f"companies/{company_name}/{file}"
                cmp_blob = bucket.blob(cmp_blob_name)
                cmp_generation_match_precondition = None
                cmp_blob.reload()
                cmp_generation_match_precondition = cmp_blob.generation
                cmp_blob.delete(if_generation_match=cmp_generation_match_precondition)

            # remove company tenant in vector db
            client_wv = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_URL"),
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            )
            mt_col_cmp = client_wv.collections.get("Companies")
            company_name_in_vector_db = change_company_name_gsc_to_vectordb(company_name=company_name)
            company_tnt = mt_col_cmp.with_tenant(company_name_in_vector_db)
            company_tnt.data.delete_many(
                where=wq.Filter.any_of([
                    wq.Filter.by_property("source_document").like(f"/{file_name}")  for file_name in company_files
                ]),
            )
            client_wv.close()
            return 0, None, f"File(s) removed."
        else:
            return 0, dash.no_update, "Please select a company and a file to remove.",
    else:
        raise dash.exceptions.PreventUpdate


# remove company folder recursively on click
@callback(
    [Output("remove_company_btn", "n_clicks"),
     Output("remove_company_loading", "children"),
     Output("remove_company_success", "children")],
    [Input("remove_company_btn", "n_clicks"),
     Input("company_dropdown", "value")],
    prevent_initial_call=True
)
def remove_company_in_gcs_and_vector_db(num_clicks, company_name):
    if num_clicks > 0:
        if company_name is not None:
            # remove company folder in gcs
            storage_client = storage.Client(credentials=gcloud_connect_credentials())
            # blobs under companies
            cmp_blobs = storage_client.list_blobs(os.getenv('GCLOUD_BUCKET_NAME'), prefix=f"companies/{company_name}/")
            # blobs under generated
            gen_blobs = storage_client.list_blobs(os.getenv('GCLOUD_BUCKET_NAME'), prefix=f"generated/{company_name}/")

            for blob in cmp_blobs:
                generation_match_precondition = None
                blob.reload()
                generation_match_precondition = blob.generation
                blob.delete(
                    if_generation_match=generation_match_precondition)

            for blob in gen_blobs:
                generation_match_precondition = None
                blob.reload()
                generation_match_precondition = blob.generation
                blob.delete(if_generation_match=generation_match_precondition)
            # also remove the empty folder in gcs
            delete_gcs_folder(
                credentials=gcloud_connect_credentials(), bucket_name=os.getenv('GCLOUD_BUCKET_NAME'),
                folder_name=f"companies/{company_name}/"
            )
            delete_gcs_folder(
                credentials=gcloud_connect_credentials(), bucket_name=os.getenv('GCLOUD_BUCKET_NAME'),
                folder_name=f"generated/{company_name}/"
            )
            # remove company tenant in vector db
            client_wv = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_URL"),
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            )
            mt_col_cmp = client_wv.collections.get("Companies")
            company_name_in_vector_db = change_company_name_gsc_to_vectordb(company_name=company_name)
            mt_col_cmp.tenants.remove([company_name_in_vector_db])
            client_wv.close()
            return 0, None, f"'{company_name}' and its files removed."
        else:
            return 0, dash.no_update, "Please select a company to remove.",
    else:
        raise dash.exceptions.PreventUpdate


# update company list dropdown according to company selected
@callback(
    Output("company_dropdown", "options"),
    [Input("add_company_success", "children"),
     Input("remove_company_success", "children")]
)
def update_company_list_dropdown(add_company_msg, remove_company_msg):
    company_list = get_company_list()
    return company_list


# update files dropdown according to company selected
@callback(
    Output("files_dropdown", "options"),
    Input("company_dropdown", "value"),
)
def update_company_files_dropdown(company_name):
    if company_name is not None:
        # list all files in a company folder
        storage_client = storage.Client(credentials=gcloud_connect_credentials())
        file_blobs = storage_client.list_blobs(
            os.getenv('GCLOUD_BUCKET_NAME'), prefix='companies/' + company_name + '/'
        )
        files_list = [
            os.path.basename(os.path.normpath(blob.name)) for blob in file_blobs
            if blob.name != 'companies/' and blob.name != 'companies/' + company_name + '/'
               and not os.path.basename(os.path.normpath(blob.name)).startswith('.')
        ]
        return files_list
    else:
        return dash.no_update


# add new company folder to gcs and vector db new tenant on Add New click
@callback(
    [Output("add_company_btn", "n_clicks"),
     Output("new_company_name", "value"),
     Output("add_company_loading", "children"),
     Output("add_company_success", "children")],
    [Input("add_company_btn", "n_clicks"),
     Input("new_company_name", "value")],
    prevent_initial_call=True
)
def add_new_company_folder(num_clicks, new_company_folder_name):
    if num_clicks > 0:
        if new_company_folder_name is not None:
            _ = create_gcs_folder(
                credentials=gcloud_connect_credentials(), bucket_name=os.getenv('GCLOUD_BUCKET_NAME'),
                folder_id="companies/" + new_company_folder_name + "/", recursive=True
            )
            new_gsc_folder_name = create_gcs_folder(
                credentials=gcloud_connect_credentials(), bucket_name=os.getenv('GCLOUD_BUCKET_NAME'),
                folder_id="generated/" + new_company_folder_name + "/", recursive=True
            )
            gsc_folder_created = os.path.basename(os.path.normpath(new_gsc_folder_name))
            # create new tenant in weaviate
            client_wv = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_URL"), auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            )
            mt_col_cmp = client_wv.collections.get("Companies")
            company_name_vector_db = change_company_name_gsc_to_vectordb(
                new_company_folder_name)
            mt_col_cmp.tenants.create(tenants=[Tenant(name=company_name_vector_db)])
            client_wv.close()
            return 0, "", None, f"Created new folder '{gsc_folder_created}'!"
        else:
            return 0, dash.no_update, dash.no_update, dash.no_update
    else:
        raise dash.exceptions.PreventUpdate


# provide current company name to chatbot system prompt
@callback(
    [Output("init_chatbot_button", component_property="n_clicks"),
     Output("init_chatbot_success", "children"),
     Output("conversation_store", "data")],
    [Input("init_chatbot_button", component_property="n_clicks"),
     Input("company_dropdown", "value")],
    prevent_initial_call=True
)
def parse_company_name_to_chatbot_system_user_prompt(num_clicks, company_name):
    if num_clicks > 0:
        # load prompts
        prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
        chatbot_system_prompt_file_name = "0_chatbot_system_prompt.txt"
        chatbot_init_user_prompt_file_name = "0_chatbot_user_prompt.txt"
        with open(prompts_path.joinpath(chatbot_system_prompt_file_name), "r") as f:
            chatbot_system_prompt = f.read()
        with open(prompts_path.joinpath(chatbot_init_user_prompt_file_name), "r") as f:
            chatbot_init_user_prompt = f.read()

        chatbot_system_prompt = chatbot_system_prompt.replace("_COMPANY_NAME_", company_name)
        chatbot_init_user_prompt = chatbot_init_user_prompt.replace("_COMPANY_NAME_", company_name)

        chatbot.chat_history.append({
            "role": "developer",
            "content": [{"type": "input_text", "text": chatbot_system_prompt}]
        })

        _, _ = chatbot.generate_response(company_name=company_name, user_prompt=chatbot_init_user_prompt)

        return 0, dbc.Alert(children="Chatbot Initialised", color="success", style={"textAlign": "center"}), json.dumps(
            [
                msg for msg in chatbot.chat_history
                if isinstance(msg, dict) and "role" in msg.keys()
                   and (msg["role"] == "assistant" or msg["role"] == "user")
                   and len(msg["content"]) != 0
            ],
            indent=2
        )
    else:
        raise dash.exceptions.PreventUpdate


# interactive chatbot
@callback(
    [Output("chatbot_send_button", "n_clicks"),
     Output("chatbot_user_input", "value"),
     Output("chat_loading_component", "children"),
     Output("conversation_store", "data", allow_duplicate=True)],
    [Input("chatbot_send_button", "n_clicks"),
     Input("company_dropdown", "value"),
     Input("chatbot_user_input", "value")],
    prevent_initial_call=True
)
def handle_chat(num_clicks, company_name, user_prompt):
    if num_clicks > 0:
        if user_prompt:
            _, _ = chatbot.generate_response(company_name=company_name, user_prompt=user_prompt)
                # model_name="o1", reasoning_effort="high", chat_history=chatbot.chat_history,
            return 0, "", None, json.dumps(
                [
                    msg for msg in chatbot.chat_history
                    if isinstance(msg, dict) and "role" in msg.keys()
                       and (msg["role"] == "assistant" or msg["role"] == "user")
                       and len(msg["content"]) != 0
                ],
                indent=2
            )
        else:
            return 0, dash.no_update, dash.no_update, dash.no_update
    else:
        raise dash.exceptions.PreventUpdate


# update chatbot display
@callback(
    Output(component_id="display_conversation", component_property="children"),
    Input(component_id="conversation_store", component_property="data")
)
def update_display(conversation_history):
    chat_history = json.loads(conversation_history)
    return [
        render_message(text=msg["content"][0]["text"], box="human") if msg["role"] == "user"
        else render_message(text=msg["content"][0]["text"], box="AI")
        for i, msg in enumerate(chat_history[2:]) if msg["role"] == "assistant" or msg["role"] == "user"
    ]  # [2:]


# render credit report if available
@callback(
    Output("credit_report", "children"),
    Input("company_dropdown", "value")
)
def render_credit_report_from_gcs(company_name):
    collection_gsc = "generated"
    report_blob_name = f"{collection_gsc}/{company_name}/credit_report.md"

    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    gen_blobs = storage_client.list_blobs(
        os.getenv('GCLOUD_BUCKET_NAME'), prefix=f"{collection_gsc}/{company_name}/", delimiter="/"
    )
    gen_list = [blob for blob in gen_blobs if blob.name == report_blob_name]
    if len(gen_list) == 1:
        bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))
        blob_md = bucket.blob(report_blob_name)
        report_contents = blob_md.download_as_bytes()
        md_render = report_contents.decode("utf-8")
    elif len(gen_list) > 1:
        md_render = "Duplicated entries. Please check."
    else:
        md_render = "No existing credit report. Generate now."
    storage_client.close()

    return md_render


# enable parse report to vector db button when credit report is available
@callback(
    [Output("download_report_btn", "disabled"),
     Output("download_report_loading_comp", "children")],
    Input("credit_report", "children")
)
def enable_download_report_button(credit_report_md_str):
    if credit_report_md_str in ["No existing credit report. Generate now.", "Duplicated entries. Please check."]:
        return True, None
    else:
        return False, None


# enable parse report to vector db button when credit report is available
@callback(
    [Output("ingest_report_to_vector_db_btn", "disabled"),
     Output("parse_report_loading_comp", "children")],
    Input("credit_report", "children")
)
def enable_parse_to_vector_db_button(credit_report_md_str):
    if credit_report_md_str in ["No existing credit report. Generate now.", "Duplicated entries. Please check."]:
        return True, None
    else:
        return False, None


# download credit report on button click
@callback(
    [Output("download_report_btn", "n_clicks"),
     Output("download_report", "data"),
     Output("download_report_html", "data")],
    [Input("download_report_btn", "n_clicks"),
     Input("company_dropdown", "value")],
    prevent_initial_call=True
)
def download_credit_report(num_clicks, company_name):
    if num_clicks > 0:
        collection_gsc = "generated"
        # download from gcs
        storage_client = storage.Client(credentials=gcloud_connect_credentials())
        bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

        blob_report = bucket.blob(f'{collection_gsc}/{company_name}/credit_report.md')
        blob_report_html = bucket.blob(f'{collection_gsc}/{company_name}/credit_report.html')

        report_contents = blob_report.download_as_bytes()
        report_contents_html = blob_report_html.download_as_bytes()

        md_str = report_contents.decode("utf-8")
        html_str = report_contents_html.decode("utf-8")

        return (0, dict(content=md_str, filename="credit_report.md"),
                dict(content=html_str, filename="credit_report.html"))
    else:
        raise dash.exceptions.PreventUpdate


# ingest report to vector db on click
@callback(
    [Output("ingest_report_to_vector_db_btn", "n_clicks"),
     Output("parse_report_loading_comp", "children", allow_duplicate=True)],
    [Input("ingest_report_to_vector_db_btn", "n_clicks"),
     Input("company_dropdown", "value")],
    prevent_initial_call= True #'initial_duplicate'
)
def parse_credit_report_to_vector_db(num_clicks, company_name):
    if num_clicks > 0:
        ####### Check for previously existing credit report object, if exist, delete before adding new
        client_wv = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
            # headers={'X-OpenAI-Api-key': os.getenv("OPENAI_API_KEY")}
        )
        # collection name
        mt_col_name = "Companies"
        mt_col_companies = client_wv.collections.get(mt_col_name)
        # list all credit report objects
        company_name_in_vector_db = change_company_name_gsc_to_vectordb(company_name=company_name)
        mt_col_companies.tenants.create(tenants=[Tenant(name=company_name_in_vector_db)])

        company_tnt = mt_col_companies.with_tenant(company_name_in_vector_db)
        company_tnt.data.delete_many(
            where=wq.Filter.by_property("source_document").like("/credit_report.md")
        )
        # client_weaviate.close()
        ####### ingest credit report to vector db
        collection_gsc = "generated"
        storage_client = storage.Client(credentials=gcloud_connect_credentials())
        bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

        blob_rep = bucket.blob(f'{collection_gsc}/{company_name}/credit_report.md')
        gsutil_uri = 'gs://' + f"{os.getenv('GCLOUD_BUCKET_NAME')}/" + f'{blob_rep.name}'

        report_elements = partition_md(file=io.BytesIO(blob_rep.download_as_bytes()))
        image_data, table_data, text_data = extract_data_list(  #narrative_text_data, other_text_data
            file_elements=report_elements, file_path=str(gsutil_uri), file_category="generated",
        )  #image_explainer_prompt=image_explainer_prompt, table_explainer_prompt=table_explainer_prompt
        # todo: create new tenant by manage files
        ingest_all_data(
            tenant_name=company_name_in_vector_db, image_data=image_data, table_data=table_data, text_data=text_data,
            collection=mt_col_name
        )  # narrative_text_data=narrative_text_data, other_text_data=other_text_data,
        client_wv.close()
        return 0, None
    else:
        raise dash.exceptions.PreventUpdate


# generate and upload to gcs on button click
@callback(
    [Output("generate_report_button", "n_clicks"),
     Output("generate_report_loading_comp", "children"),
     Output("credit_report", "children", allow_duplicate=True)],
    [Input("generate_report_button", "n_clicks"),
     Input("company_dropdown", "value")],
    prevent_initial_call='initial_duplicate'
)
def generate_credit_report_and_upload_to_gcs(num_clicks, company_name):
    if num_clicks > 0:
        rpt = CreditReportGenerate(company_name=company_name)
        rpt.generate_credit_report()

        # upload credit report markdown, json and reasoning steps to gcs
        storage_client = storage.Client(credentials=gcloud_connect_credentials())
        bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

        blob_html = bucket.blob(f'generated/{company_name}/credit_report.html')
        blob_md = bucket.blob(f'generated/{company_name}/credit_report.md')
        blob_js = bucket.blob(f'generated/{company_name}/credit_report_dict.json')
        blob_rs = bucket.blob(f'generated/{company_name}/reasoning_steps_dict.json')

        html_str = markdown.markdown(rpt.credit_report_str, extensions=['nl2br', 'markdown.extensions.tables'])
        blob_html.upload_from_string(html_str, content_type="text/plain")
        blob_md.upload_from_string(rpt.credit_report_str, content_type="application/octet-stream")
        blob_js.upload_from_string(json.dumps(rpt.report_dct, indent=2), content_type="application/json")
        blob_rs.upload_from_string(json.dumps(rpt.reason_dct, indent=2), content_type="application/json")

        storage_client.close()
        return 0, None, rpt.credit_report_str
    else:
        raise dash.exceptions.PreventUpdate
