import time
import io
import os
import re
import pathlib
import json
import vertexai

from vertexai.generative_models import GenerativeModel, Part, Image, Content, Tool, grounding

from google import genai
from google.genai import types
from google.cloud import storage

from src.util import (
    ReasoningToolSchema, process_files_content_parts, gemini_pro_loop_with_tools, gcloud_connect_credentials
)

# load .env if not in production
if not bool(os.getenv("PRODUCTION_MODE")):
    from dotenv import load_dotenv
    load_dotenv()


### tool JSON Schema
tl = ReasoningToolSchema()


class CreditReportGenerate:
    def __init__(self, company_name):
        self.company_name = company_name
        # self.user_quant_metrics = user_quant_metrics
        self.report_dct = {}
        self.reason_dct = {}
        self.credit_report_str = ""
        self.date_and_lst_fin = self.find_date_today_and_last_financial_period()


    def find_date_today_and_last_financial_period(self):
        # get date today
        client_gm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        date_prompt_msg = (
            "Tell me today's date in Anywhere-on-Earth (AoE) timezone. Write your response in the format: "
            "DD-MONTH-YYYY. 'DD' is the two-digit for the day, 'YYYY' is the four-digit representing the year. "
            "Do NOT include any extra comments and explanation."
        )
        response_date = client_gm.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=date_prompt_msg,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="text/plain",
            )
        )
        date_today = re.sub("```.+\n|```\n|```", "", response_date.text)
        # load files classifying from gcs blob
        storage_client = storage.Client(credentials=gcloud_connect_credentials())
        bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

        blob = bucket.blob(f'generated/{self.company_name}/class_file_dict.json')
        contents = blob.download_as_bytes()
        class_file_dict = json.loads(contents.decode("utf-8"))

        # initialise vertexai
        vertexai.init(
            credentials=gcloud_connect_credentials(), project=os.getenv('GCLOUD_PROJECT_ID'), location="us-central1"
        )

        file_blobs = storage_client.list_blobs(
            os.getenv('GCLOUD_BUCKET_NAME'), prefix='companies/' + self.company_name + '/'
        )
        file_list = [blob for blob in file_blobs if
                     blob.name != 'companies/' and blob.name != 'companies/' + self.company_name + '/']
        # load file uri from class_file_dict
        financial_file_blob_list = []
        for blob in file_list:
            gsutil_uri = 'gs://' + f"{os.getenv('GCLOUD_BUCKET_NAME')}/" + f'{blob.name}'
            if gsutil_uri in class_file_dict['financial']:
                financial_file_blob_list.append(blob)

        fin_files_content_parts_vertexai = []
        for file_blob in financial_file_blob_list:
            files_content_parts = process_files_content_parts(os.getenv('GCLOUD_BUCKET_NAME'), file_blob, vertexai=True)
            fin_files_content_parts_vertexai.extend(files_content_parts)

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 65535,
            "response_mime_type": "text/plain",
        }
        model_fin_prd = GenerativeModel(
            model_name="gemini-2.5-flash-preview-04-17",
            generation_config=generation_config,
        )
        user_prompt = (
            f"Today's date is:{date_today}. Based on all the information from the documents, identify the "
            f"Latest Financial Reporting Period that is the most recent reporting period found and CLOSEST TO "
            f"today's date. You must DOUBLE CHECK your answer on the Latest Financial "
            "Reporting Period. It is EXTREMELY IMPORTANT to get the Latest Financial Reporting Period CORRECT, "
            "because another human's work depend completely on your response for later work. A wrong answer from you "
            "on the Latest Financial Reporting Period will make the person face SERIOUS PENALTY! Write your response "
            "in the following exact format, WITHOUT any extra comments: \n"
            "[Date in the form of Day-Month-Year, where Month is spelled in full, Year displayed in 4 digits] "
            "([<H1 or H2> <Year>, where H1 or H2 represents the half-year that is CLOSEST AFTER the date of "
            "latest financial reporting period you identified. It is the closest half-year NEXT TO and AFTER the "
            "Latest Financial Reporting Period.])"
        )
        fin_files_contents_prompt = [
            Content(
                role="user",
                parts=fin_files_content_parts_vertexai + [Part.from_text(user_prompt)]
            )
        ]
        response_lst_fin = model_fin_prd.generate_content(contents=fin_files_contents_prompt)
        last_fin_period = re.sub("```.+\n|```\n|```", "", response_lst_fin.text)

        date_lst_fin = (
            f"Today's date in Anywhere-on-Earth (AoE) timezone that must be used as the CUTOFF DATE for analysis "
            f"and data knowledge is: {date_today}, and the Latest Financial Reporting Period of the company "
            f"{self.company_name} according to the documents we have is: {last_fin_period}."
        )
        return date_lst_fin


    def find_date_today_and_latest_fin_period_o1(self):
        user_prompt = (
            "Identify for today's date in Anywhere-on-Earth (AoE) timezone using web search results, and make "
            "today's date as the CUTOFF DATE for analysis and data knowledge. Based on all the information from the "
            "RAG knowledge vector database, retrieve all the reporting period information available, "
            "and identify the Latest Financial Reporting Period that is the most recent reporting period found "
            "closest to today's date. You must DOUBLE CHECK your answer on the Latest Financial Reporting Period "
            "on web search results. It is EXTREMELY IMPORTANT to get the Latest Financial Reporting Period CORRECT, "
            "because another human's work depend completely on your response for later work. A wrong answer from you "
            "on the Latest Financial Reporting Period will make the person face serious penalty! Write the response "
            "in the following exact format, WITHOUT any extra comments: \n"
            "Today's date in Anywhere-on-Earth (AoE) timezone that must be used as the CUTOFF DATE for analysis "
            "and data knowledge is: [Date in the form of Day-Month-Year, where Month is spelled in full, Year "
            f"displayed in 4 digits], and the Latest Financial Reporting Period of the company {self.company_name} "
            f"according to the documents we have is: "
            "[Date in the form of Day-Month-Year, where Month is spelled in full, Year displayed in 4 digits] "
            "([<H1 or H2> <Year>, for example: if last financial reporting period is 31 December 2021, "
            "then it should be 'H1 2022'. For example: if last financial reporting period is 30 June 2018, "
            "then it should be 'H2 2018'])"
        )
        initial_input_messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}]
            }
        ]
        response, _ = gemini_pro_loop_with_tools(
            model_name="o1", reasoning_effort="high", input_messages=initial_input_messages,
            tools=[tl.gemini_web_search_tool, tl.rag_retrieval_tool],
            file_category=["ratings", "financial", "bond", "additional"],
            company_name=self.company_name
        )
        return re.sub("```.+\n|```\n|```", "", response)


    def generate_credit_report(self):
        print("Generating Section 2: Business Model Overview")
        _, _ = self.generate_business_overview()
        self.credit_report_str += self.report_dct["business_overview"]

        print("Generating Section 3: Financial Health")
        _, _ = self.generate_financial_health()
        self.credit_report_str += "\n\n" + self.report_dct["financial_health"]

        print("Generating Section 4: Debt Structure & Maturity Profile")
        _, _ = self.generate_debt_structure()
        self.credit_report_str += "\n\n" + self.report_dct["debt_structure"]

        print("Generating Section 5: Credit Risk")
        _, _ = self.generate_credit_risk()
        self.credit_report_str += "\n\n" + self.report_dct["credit_risk"]

        print("Generating Section 1: Executive Summary")
        _, _ = self.generate_executive_summary()
        self.credit_report_str = self.report_dct["exec_summary"] + "\n\n" + self.credit_report_str


    def join_report_section_all(self):
        self.credit_report_str += self.report_dct["exec_summary"]
        self.credit_report_str += "\n\n" + self.report_dct["business_overview"]
        self.credit_report_str += "\n\n" + self.report_dct["financial_health"]
        self.credit_report_str += "\n\n" + self.report_dct["debt_structure"]
        self.credit_report_str += "\n\n" + self.report_dct["credit_risk"]


    ####### section: executive summary
    def generate_executive_summary(self):
        # load prompt
        prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
        exec_summary_prompt_file_name = "1_executive_summary.txt"
        with open(prompts_path.joinpath(exec_summary_prompt_file_name), "r") as f:
            exec_summary_prompt = f.read()
        exec_summary_prompt = (exec_summary_prompt.replace("_COMPANY_NAME_", self.company_name)
                               .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", self.date_and_lst_fin))

        credit_report_md_str_no_summary = (
                self.report_dct['business_overview'] + '\n\n' + self.report_dct['financial_health'] + '\n\n'
                + self.report_dct['debt_structure'] + '\n\n' + self.report_dct['credit_risk']
        )
        initial_input_messages = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": exec_summary_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Remaining section of the credit report in markdown:\n{credit_report_md_str_no_summary}"
                    }
                ]
            },
        ]
        summary_response, summary_step_outputs = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages,
            tools=[tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_tool],
            file_category=["financial", "ratings"]
        )
        # add to class dict
        section_str = re.sub("```.+\n|```\n|```", "", summary_response)
        self.report_dct["exec_summary"] = section_str
        self.reason_dct["exec_summary"] = summary_step_outputs
        return section_str, summary_step_outputs


    ####### section 2: Business model overview
    def generate_business_overview(self):
        # load business overview generation prompts
        prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
        business_overview_prompt_file_name = "2_business_overview.txt"
        with open(prompts_path.joinpath(business_overview_prompt_file_name), "r") as f:
            business_overview_prompt = f.read()
        # give the exact company name
        business_overview_prompt = (
            business_overview_prompt.replace("_COMPANY_NAME_", self.company_name)
            .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", self.date_and_lst_fin)
        )
        initial_input_messages = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": business_overview_prompt}]
            }
        ]
        tools = [
            tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_tool,
        ]
        response, step_outputs = gemini_pro_loop_with_tools(
            model_name="o1", reasoning_effort="high", input_messages=initial_input_messages, tools=tools,
            file_category=["ratings", "financial"],
            company_name=self.company_name
        )
        section_str = re.sub("```.+\n|```\n|```", "", response)

        self.report_dct["business_overview"] = section_str
        self.reason_dct["business_overview"] = step_outputs
        return section_str, step_outputs


    ####### section: financial health
    def generate_financial_health(self):
        # load prompts
        # MAIN_PATH = pathlib.Path(__file__).parent
        # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
        prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
        fin_system_prompt_file_name = "3_financial_health_system_prompt.txt"
        fin_user_prompt_1_metric_file_name = "3_financial_health_user_prompt_1_metric.txt"
        # fin_user_prompt_1_user_metric_file_name = "3_financial_health_user_prompt_1_user_quant_table.txt"
        fin_user_prompt_2_analyse_file_name = "3_financial_health_user_prompt_2_analyse.txt"

        with open(prompts_path.joinpath(fin_system_prompt_file_name), "r") as f:
            fin_system_prompt = f.read()

        fin_system_prompt = (
            fin_system_prompt.replace("_COMPANY_NAME_", self.company_name)
            .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", self.date_and_lst_fin)
        )

        with open(prompts_path.joinpath(fin_user_prompt_1_metric_file_name), "r") as f:
            fin_user_prompt_1_metric = f.read()
        with open(prompts_path.joinpath(fin_user_prompt_2_analyse_file_name), "r") as f:
            fin_user_prompt_2_analyse = f.read()

        initial_input_messages_1 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": fin_system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": fin_user_prompt_1_metric}]
            },
        ]
        tools = [
            tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_for_table_tool,
        ]
        # Step 1: create metric tables
        fin_health_response_1, fin_health_step_outputs_1 = gemini_pro_loop_with_tools(
            model_name="o1", reasoning_effort="high", input_messages=initial_input_messages_1, tools=tools,
            file_category=["financial", "ratings", "additional"], company_name=self.company_name
        )
        # Step 2: write analysis
        initial_input_messages_2 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": fin_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": fin_user_prompt_2_analyse},
                    {"type": "input_text", "text": f"Markdown output from Step 1: \n{fin_health_response_1}"}
                ]
            },
        ]
        fin_health_response_2, fin_health_step_outputs_2 = gemini_pro_loop_with_tools(
            model_name="o1", reasoning_effort="high", input_messages=initial_input_messages_2, tools=tools,
            file_category=["financial", "ratings"], company_name=self.company_name
        )
        fin_health_response = (
                re.sub("```.+\n|```\n|```", "", fin_health_response_1) + "\n\n"
                + re.sub("```.+\n|```\n|```", "", fin_health_response_2)
        )
        fin_health_step_outputs = fin_health_step_outputs_1 + fin_health_step_outputs_2

        self.report_dct["financial_health"] = fin_health_response
        self.reason_dct["financial_health"] = fin_health_step_outputs
        return fin_health_response, fin_health_step_outputs


    def generate_debt_structure(self):
        # load prompts
        prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
        system_prompt_file_name = "4_debt_structure_system_prompt.txt"
        user_prompt_1_table1_bond_file_name = "4_debt_structure_user_prompt_1_table1_bond.txt"
        user_prompt_1_table2_g3_file_name = "4_debt_structure_user_prompt_1_table2_g3.txt"
        user_prompt_1_table3_local_file_name = "4_debt_structure_user_prompt_1_table3_local.txt"
        user_prompt_1_table4_maturity_file_name = "4_debt_structure_user_prompt_1_table4_maturity.txt"
        user_prompt_2_analyse_file_name = "4_debt_structure_user_prompt_2_writing.txt"

        with open(prompts_path.joinpath(system_prompt_file_name), "r") as f:
            system_prompt = f.read()

        system_prompt = (
            system_prompt.replace("_COMPANY_NAME_", self.company_name)
            .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", self.date_and_lst_fin)
        )

        with open(prompts_path.joinpath(user_prompt_1_table1_bond_file_name), "r") as f:
            user_prompt_1_table1_bond = f.read()
        with open(prompts_path.joinpath(user_prompt_1_table2_g3_file_name), "r") as f:
            user_prompt_1_table2_g3 = f.read()
        with open(prompts_path.joinpath(user_prompt_1_table3_local_file_name), "r") as f:
            user_prompt_1_table3_local = f.read()
        with open(prompts_path.joinpath(user_prompt_1_table4_maturity_file_name), "r") as f:
            user_prompt_1_table4_maturity = f.read()
        with open(prompts_path.joinpath(user_prompt_2_analyse_file_name), "r") as f:
            user_prompt_2_analyse = f.read()

        debt_tables_str = ""
        debt_step_outputs = []
        tools = [tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_bonds_api_for_chatbot_tool]
        # Step 1: create table for bond series
        initial_input_messages_1 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt_1_table1_bond}]
            },
        ]
        response_1, step_outputs_1 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_1,
            tools=tools,
            file_category=["ratings", "bond"]
        )
        debt_tables_str += re.sub("```.+\n|```\n|```", "", response_1)
        debt_step_outputs.extend(step_outputs_1)
        # Step 2: create table for g3 debt
        initial_input_messages_2 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt_1_table2_g3},
                    {"type": "input_text", "text": f"Bond table Markdown output from Step 1: \n{debt_tables_str}"}
                ]
            },
        ]
        response_2, step_outputs_2 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_2,
            tools=tools,
            file_category=["ratings", "bond"]
        )
        debt_tables_str += "\n\n" + re.sub("```.+\n|```\n|```", "", response_2)
        debt_step_outputs.extend(step_outputs_2)
        # Step 3: create table for local debt
        initial_input_messages_3 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt_1_table3_local},
                    {"type": "input_text",
                     "text": f"Debt and bond table Markdown output from previous steps: \n{debt_tables_str}"}
                ]
            },
        ]
        response_3, step_outputs_3 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_3,
            tools=[tl.rag_retrieval_tool, tl.gemini_web_search_tool],
            file_category=["ratings", "bond"]
        )
        debt_tables_str += "\n\n" + re.sub("```.+\n|```\n|```", "", response_3)
        debt_step_outputs.extend(step_outputs_3)
        # Step 4: create table for debt maturity wall
        initial_input_messages_4 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt_1_table4_maturity},
                    {"type": "input_text",
                     "text": f"Debt and bond table Markdown output from previous steps: \n{debt_tables_str}"}
                ]
            },
        ]
        response_4, step_outputs_4 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_4,
            tools=tools,
            file_category=["ratings", "bond"]
        )
        debt_tables_str += "\n\n" + re.sub("```.+\n|```\n|```", "", response_4)
        debt_step_outputs.extend(step_outputs_4)
        # Step 5: write analysis
        initial_input_messages_analyse = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt_2_analyse},
                    {"type": "input_text",
                     "text": f"Debt and bond markdown output from previous steps: \n{debt_tables_str}"}
                ]
            },
        ]
        response_analyse, step_outputs_analyse = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_analyse,
            tools=tools,
            file_category=["ratings", "bond"]
        )
        # separate out the "### Debt Instrument Summaries" section
        debt_analyse_str_ls = re.sub("```.+\n|```\n|```", "", response_analyse).split("### Debt Instrument Summaries")
        debt_intro_str = "\n\n".join(debt_analyse_str_ls[:-1])
        debt_inst_sum_str = "\n\n### Debt Instrument Summaries" + debt_analyse_str_ls[-1]
        section_response = debt_intro_str + "\n\n" + debt_tables_str + debt_inst_sum_str
        section_step_outputs = step_outputs_analyse + debt_step_outputs

        self.report_dct["debt_structure"] = section_response
        self.reason_dct["debt_structure"] = section_step_outputs
        return section_response, section_step_outputs


    ####### section: credit risk
    def generate_credit_risk(self):
        # MAIN_PATH = pathlib.Path(__file__).parent
        # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
        prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
        cred_system_prompt_file_name = "5_credit_risk_system_prompt.txt"
        cred_user_prompt_1_rtgs_file_name = "5_credit_risk_user_prompt_1_ratings.txt"
        cred_user_prompt_2_fin_file_name = "5_credit_risk_user_prompt_2_financial.txt"
        cred_user_prompt_3_bond_file_name = "5_credit_risk_user_prompt_3_bond.txt"
        cred_user_prompt_4_add_file_name = "5_credit_risk_user_prompt_4_additional.txt"
        cred_user_prompt_5_web_file_name = "5_credit_risk_user_prompt_5_web.txt"
        with open(prompts_path.joinpath(cred_system_prompt_file_name), "r") as f:
            cred_system_prompt = f.read()
        with open(prompts_path.joinpath(cred_user_prompt_1_rtgs_file_name), "r") as f:
            cred_user_prompt_1_rtgs = f.read()
        with open(prompts_path.joinpath(cred_user_prompt_2_fin_file_name), "r") as f:
            cred_user_prompt_2_fin = f.read()
        with open(prompts_path.joinpath(cred_user_prompt_3_bond_file_name), "r") as f:
            cred_user_prompt_3_bond = f.read()
        with open(prompts_path.joinpath(cred_user_prompt_4_add_file_name), "r") as f:
            cred_user_prompt_4_add = f.read()
        with open(prompts_path.joinpath(cred_user_prompt_5_web_file_name), "r") as f:
            cred_user_prompt_5_web = f.read()

        cred_system_prompt = (
            cred_system_prompt.replace("_COMPANY_NAME_", self.company_name)
            .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", self.date_and_lst_fin)
        )
        cred_user_prompt_1_rtgs = cred_user_prompt_1_rtgs.replace("_COMPANY_NAME_", self.company_name)
        cred_user_prompt_2_fin = cred_user_prompt_2_fin.replace("_COMPANY_NAME_", self.company_name)
        cred_user_prompt_3_bond = cred_user_prompt_3_bond.replace("_COMPANY_NAME_", self.company_name)
        cred_user_prompt_4_add = cred_user_prompt_4_add.replace("_COMPANY_NAME_", self.company_name)
        cred_user_prompt_5_web = cred_user_prompt_5_web.replace("_COMPANY_NAME_", self.company_name)

        cred_step_outputs = []
        tools = [tl.rag_retrieval_tool, tl.get_financial_metrics_yf_api_tool]
        # Step 1: over ratings bucket
        print("Generating Section 5: Credit Risk... Step 1: Ratings bucket")
        initial_input_messages_1 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": cred_system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": cred_user_prompt_1_rtgs}]
            },
        ]
        cred_response, cred_step_outputs_1 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_1,
            tools=tools,
            file_category=["ratings"]
        )
        cred_step_outputs.extend(cred_step_outputs_1)
        # Step 2: from financial bucket
        print("Generating Section 5: Credit Risk... Step 2: Financial bucket")
        initial_input_messages_2 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": cred_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": cred_user_prompt_2_fin},
                    {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 1: \n{cred_response}"}
                ]
            },
        ]
        cred_response, cred_step_outputs_2 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_2,
            tools=tools,
            file_category=["financial"]
        )
        cred_step_outputs.extend(cred_step_outputs_2)
        # Step 3: from bond bucket
        print("Generating Section 5: Credit Risk... Step 3: Bond bucket")
        initial_input_messages_3 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": cred_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": cred_user_prompt_3_bond},
                    {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 2: \n{cred_response}"}
                ]
            },
        ]
        cred_response, cred_step_outputs_3 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_3,
            tools=tools,
            file_category=["bond"]
        )
        cred_step_outputs.extend(cred_step_outputs_3)
        # Step 4: from additional bucket
        print("Generating Section 5: Credit Risk... Step 4: Additional bucket")
        initial_input_messages_4 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": cred_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": cred_user_prompt_4_add},
                    {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 3: \n{cred_response}"}
                ]
            },
        ]
        cred_response, cred_step_outputs_4 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_4,
            tools=tools,
            file_category=["additional"]
        )
        cred_step_outputs.extend(cred_step_outputs_4)
        # Step 5: from web search
        print("Generating Section 5: Credit Risk... Step 5: Web search")
        initial_input_messages_5 = [
            {
                "role": "model",
                "content": [{"type": "input_text", "text": cred_system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": cred_user_prompt_5_web},
                    {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 3: \n{cred_response}"}
                ]
            },
        ]
        cred_response, cred_step_outputs_5 = gemini_pro_loop_with_tools(
            model_name="o1",
            reasoning_effort="high",
            input_messages=initial_input_messages_5,
            tools=[tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_tool],
            file_category=None
        )
        cred_step_outputs.extend(cred_step_outputs_4)
        section_str = re.sub("```.+\n|```\n|```", "", cred_response)

        self.report_dct["credit_risk"] = section_str
        self.reason_dct["credit_risk"] = cred_step_outputs
        return section_str, cred_step_outputs



def generate_credit_report(company_name, credit_report_dict, reasoning_steps_dict):
    print("Generating Latest Financial Reporting Period")
    date_and_lst_fin, reason_lst = find_date_today_and_latest_fin_period(company_name=company_name)
    # time.sleep(10)
    print("Generating Section 2: Business Model Overview")
    # response_bus, reason_bus = "Test response for section 2, business model overview", ["Test reasoning for section 2"]
    response_bus, reason_bus = generate_business_overview(
        company_name=company_name, date_and_latest_fin=date_and_lst_fin
    )
    credit_report_dict["business_overview"] = response_bus
    reasoning_steps_dict["business_overview"] = reason_bus

    print("Generating Section 3: Financial Health")
    # response_fin, reason_fin = "Test response for section 3, financial health", ["Test reasoning for section 3"]
    response_fin, reason_fin = generate_financial_health(
        company_name=company_name, date_and_latest_fin=date_and_lst_fin
    )
    credit_report_dict["financial_health"] = response_fin
    reasoning_steps_dict["financial_health"] = reason_fin

    print("Generating Section 4: Debt Structure & Maturity Profile")
    # response_debt, reason_debt = "Test response for section 4, debt structure", ["Test reasoning for section 4"]
    response_debt, reason_debt = generate_debt_structure(
        company_name=company_name, date_and_latest_fin=date_and_lst_fin
    )
    credit_report_dict["debt_structure"] = response_debt
    reasoning_steps_dict["debt_structure"] = reason_debt

    print("Generating Section 5: Credit Risk")
    # response_cred, reason_cred = "Test response for section 5, credit risk", ["Test reasoning for section 5"]
    response_cred, reason_cred = generate_credit_risk(company_name=company_name, date_and_latest_fin=date_and_lst_fin)
    credit_report_dict["credit_risk"] = response_cred
    reasoning_steps_dict["credit_risk"] = reason_cred

    print("Generating Section 1: Executive Summary")
    # response_sum, reason_sum = "Test response for section 1, executive summary", ["Test reasoning for section 1"]
    response_sum, reason_sum = generate_executive_summary(
        company_name=company_name, date_and_latest_fin=date_and_lst_fin, credit_report_dict=credit_report_dict
    )
    credit_report_dict["exec_summary"] = response_sum
    reasoning_steps_dict["exec_summary"] = reason_sum

    return credit_report_dict, reasoning_steps_dict


def find_date_today_and_latest_fin_period(company_name):
    user_prompt = (
        "Identify for today's date in Anywhere-on-Earth (AoE) timezone using web search results, and make today's date "
        "as the CUTOFF DATE for analysis and data knowledge. Based on all the information from the "
        "RAG knowledge vector database, identify the Latest Financial Reporting Period, and write the response "
        "in the following exact format, WITHOUT any extra comments: \n"
        "Today's date in Anywhere-on-Earth (AoE) timezone that must be used as the CUTOFF DATE for analysis "
        "and data knowledge is: [Date in the form of Day-Month-Year, where Month is spelled in full, "
        "Year displayed in 4 digits], and the Latest Financial Reporting Period according to the documents we have is: "
        "[Date in the form of Day-Month-Year, where Month is spelled in full, Year displayed in 4 digits] "
        "([H1 or H2] [Year])"
    )
    initial_input_messages = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}]
        }
    ]
    response, step_outputs = gemini_pro_loop_with_tools(
        model_name="o1", reasoning_effort="high", input_messages=initial_input_messages,
        tools=[tl.gemini_web_search_tool, tl.rag_retrieval_tool],
        file_category=["ratings", "financial", "bond", "additional"],
        company_name=company_name
    )
    return re.sub("```.+\n|```\n|```", "", response), step_outputs


def generate_business_overview(company_name, date_and_latest_fin):
    ### tool JSON Schema
    # tl = ReasoningToolSchema()
    ####### section 2: Business model overview
    # load business overview generation prompts
    prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
    business_overview_prompt_file_name = "2_business_overview.txt"
    with open(prompts_path.joinpath(business_overview_prompt_file_name), "r") as f:
        business_overview_prompt = f.read()
    # give the exact company name
    business_overview_prompt = (business_overview_prompt.replace("_COMPANY_NAME_", company_name)
                                .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", date_and_latest_fin))
    initial_input_messages = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": business_overview_prompt}]
        }
    ]
    tools = [
        tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_tool,
        tl.get_financial_metrics_api_for_chatbot_tool
    ]
    response, step_outputs = gemini_pro_loop_with_tools(
        model_name="o1", reasoning_effort="high", input_messages=initial_input_messages, tools=tools,
        file_category=["ratings", "financial"],
        company_name=company_name
    )
    return re.sub("```.+\n|```\n|```", "", response), step_outputs


def generate_financial_health(company_name, date_and_latest_fin):
    ### tool JSON Schema
    # tl = ReasoningToolSchema()
    # load prompts
    # MAIN_PATH = pathlib.Path(__file__).parent
    # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
    prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
    fin_system_prompt_file_name = "3_financial_health_system_prompt.txt"
    fin_user_prompt_1_metric_file_name = "3_financial_health_user_prompt_1_metric.txt"
    # fin_user_prompt_1_user_metric_file_name = "3_financial_health_user_prompt_1_user_quant_table.txt"
    fin_user_prompt_2_analyse_file_name = "3_financial_health_user_prompt_2_analyse.txt"

    with open(prompts_path.joinpath(fin_system_prompt_file_name), "r") as f:
        fin_system_prompt = f.read()

    fin_system_prompt = (fin_system_prompt.replace("_COMPANY_NAME_", company_name)
                         .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", date_and_latest_fin))

    with open(prompts_path.joinpath(fin_user_prompt_1_metric_file_name), "r") as f:
        fin_user_prompt_1_metric = f.read()
    with open(prompts_path.joinpath(fin_user_prompt_2_analyse_file_name), "r") as f:
        fin_user_prompt_2_analyse = f.read()

    initial_input_messages_1 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": fin_system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": fin_user_prompt_1_metric}]
        },
    ]
    tools = [
        tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_for_table_tool,
        # tl.get_financial_metrics_api_for_chatbot_tool
    ]
    # Step 1: create metric tables
    fin_health_response_1, fin_health_step_outputs_1 = gemini_pro_loop_with_tools(
        model_name="o1", reasoning_effort="high", input_messages=initial_input_messages_1, tools=tools,
        file_category=["financial", "ratings"], company_name=company_name
    )
    # Step 2: write analysis
    initial_input_messages_2 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": fin_system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": fin_user_prompt_2_analyse},
                {"type": "input_text", "text": f"Markdown output from Step 1: \n{fin_health_response_1}"}
            ]
        },
    ]
    fin_health_response_2, fin_health_step_outputs_2 = gemini_pro_loop_with_tools(
        model_name="o1", reasoning_effort="high", input_messages=initial_input_messages_2, tools=tools,
        file_category=["financial", "ratings"], company_name=company_name
    )
    fin_health_response = (
            re.sub("```.+\n|```\n|```", "", fin_health_response_1) + "\n\n"
            + re.sub("```.+\n|```\n|```", "", fin_health_response_2)
    )
    fin_health_step_outputs = fin_health_step_outputs_1 + fin_health_step_outputs_2
    return fin_health_response, fin_health_step_outputs


def generate_debt_structure(company_name, date_and_latest_fin):

    # load prompts
    prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
    system_prompt_file_name = "4_debt_structure_system_prompt.txt"
    user_prompt_1_table1_bond_file_name = "4_debt_structure_user_prompt_1_table1_bond.txt"
    user_prompt_1_table2_g3_file_name = "4_debt_structure_user_prompt_1_table2_g3.txt"
    user_prompt_1_table3_local_file_name = "4_debt_structure_user_prompt_1_table3_local.txt"
    user_prompt_1_table4_maturity_file_name = "4_debt_structure_user_prompt_1_table4_maturity.txt"
    user_prompt_2_analyse_file_name = "4_debt_structure_user_prompt_2_writing.txt"

    with open(prompts_path.joinpath(system_prompt_file_name), "r") as f:
        system_prompt = f.read()

    system_prompt = (system_prompt.replace("_COMPANY_NAME_", company_name)
                     .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", date_and_latest_fin))

    with open(prompts_path.joinpath(user_prompt_1_table1_bond_file_name), "r") as f:
        user_prompt_1_table1_bond = f.read()
    with open(prompts_path.joinpath(user_prompt_1_table2_g3_file_name), "r") as f:
        user_prompt_1_table2_g3 = f.read()
    with open(prompts_path.joinpath(user_prompt_1_table3_local_file_name), "r") as f:
        user_prompt_1_table3_local = f.read()
    with open(prompts_path.joinpath(user_prompt_1_table4_maturity_file_name), "r") as f:
        user_prompt_1_table4_maturity = f.read()
    with open(prompts_path.joinpath(user_prompt_2_analyse_file_name), "r") as f:
        user_prompt_2_analyse = f.read()

    debt_tables_str = ""
    debt_step_outputs = []
    tools = [tl.rag_retrieval_tool, tl.gemini_web_search_tool, tl.get_bonds_api_for_chatbot_tool]
    # Step 1: create table for bond series
    initial_input_messages_1 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt_1_table1_bond}]
        },
    ]
    response_1, step_outputs_1 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_1,
        tools=tools,
        file_category=["ratings", "bond"]
    )
    debt_tables_str += re.sub("```.+\n|```\n|```", "", response_1)
    debt_step_outputs.extend(step_outputs_1)
    # Step 2: create table for g3 debt
    initial_input_messages_2 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt_1_table2_g3},
                {"type": "input_text", "text": f"Bond table Markdown output from Step 1: \n{debt_tables_str}"}
            ]
        },
    ]
    response_2, step_outputs_2 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_2,
        tools=tools,
        file_category=["ratings", "bond"]
    )
    debt_tables_str += "\n\n" + re.sub("```.+\n|```\n|```", "", response_2)
    debt_step_outputs.extend(step_outputs_2)
    # Step 3: create table for local debt
    initial_input_messages_3 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt_1_table3_local},
                {"type": "input_text", "text": f"Debt and bond table Markdown output from previous steps: \n{debt_tables_str}"}
            ]
        },
    ]
    response_3, step_outputs_3 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_3,
        tools=[tl.rag_retrieval_tool, tl.gemini_web_search_tool],
        file_category=["ratings", "bond"]
    )
    debt_tables_str += "\n\n" + re.sub("```.+\n|```\n|```", "", response_3)
    debt_step_outputs.extend(step_outputs_3)
    # Step 4: create table for debt maturity wall
    initial_input_messages_4 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt_1_table4_maturity},
                {"type": "input_text",
                 "text": f"Debt and bond table Markdown output from previous steps: \n{debt_tables_str}"}
            ]
        },
    ]
    response_4, step_outputs_4 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_4,
        tools=tools,
        file_category=["ratings", "bond"]
    )
    debt_tables_str += "\n\n" + re.sub("```.+\n|```\n|```", "", response_4)
    debt_step_outputs.extend(step_outputs_4)
    # Step 5: write analysis
    initial_input_messages_analyse = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt_2_analyse},
                {"type": "input_text", "text": f"Debt and bond markdown output from previous steps: \n{debt_tables_str}"}
            ]
        },
    ]
    response_analyse, step_outputs_analyse = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_analyse,
        tools=tools,
        file_category=["ratings", "bond"]
    )
    # separate out the "### Debt Instrument Summaries" section
    debt_analyse_str_ls = re.sub("```.+\n|```\n|```", "", response_analyse).split("### Debt Instrument Summaries")
    debt_intro_str = "\n\n".join(debt_analyse_str_ls[:-1])
    debt_inst_sum_str = "\n\n### Debt Instrument Summaries" + debt_analyse_str_ls[-1]
    section_response = debt_intro_str + "\n\n" + debt_tables_str + debt_inst_sum_str
    section_step_outputs = step_outputs_analyse + debt_step_outputs
    return section_response, section_step_outputs


def generate_credit_risk(company_name, date_and_latest_fin):
    ### tool JSON Schema
    # tl = ReasoningToolSchema()
    # MAIN_PATH = pathlib.Path(__file__).parent
    # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
    prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
    cred_system_prompt_file_name = "5_credit_risk_system_prompt.txt"
    cred_user_prompt_1_ratings_file_name = "5_credit_risk_user_prompt_1_ratings.txt"
    cred_user_prompt_2_fin_file_name = "5_credit_risk_user_prompt_2_financial.txt"
    cred_user_prompt_3_add_file_name = "5_credit_risk_user_prompt_3_additional.txt"
    cred_user_prompt_4_web_file_name = "5_credit_risk_user_prompt_4_web.txt"
    with open(prompts_path.joinpath(cred_system_prompt_file_name), "r") as f:
        cred_system_prompt = f.read()
    with open(prompts_path.joinpath(cred_user_prompt_1_ratings_file_name), "r") as f:
        cred_user_prompt_1_ratings = f.read()
    with open(prompts_path.joinpath(cred_user_prompt_2_fin_file_name), "r") as f:
        cred_user_prompt_2_fin = f.read()
    with open(prompts_path.joinpath(cred_user_prompt_3_add_file_name), "r") as f:
        cred_user_prompt_3_add = f.read()
    with open(prompts_path.joinpath(cred_user_prompt_4_web_file_name), "r") as f:
        cred_user_prompt_4_web = f.read()

    cred_system_prompt = (cred_system_prompt.replace("_COMPANY_NAME_", company_name)
                          .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", date_and_latest_fin))
    cred_user_prompt_1_ratings = cred_user_prompt_1_ratings.replace("_COMPANY_NAME_", company_name)
    cred_user_prompt_2_fin = cred_user_prompt_2_fin.replace("_COMPANY_NAME_", company_name)
    cred_user_prompt_3_add = cred_user_prompt_3_add.replace("_COMPANY_NAME_", company_name)
    cred_user_prompt_4_web = cred_user_prompt_4_web.replace("_COMPANY_NAME_", company_name)

    cred_step_outputs = []
    tools = [
        tl.rag_retrieval_tool, tl.get_financial_metrics_yf_api_tool #, tl.get_financial_metrics_api_for_chatbot_tool
        #tl.get_bonds_api_for_chatbot_tool,
    ]
    # Step 1: over ratings bucket
    print("Generating Section 5: Credit Risk... Step 1: Ratings bucket")
    initial_input_messages_1 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": cred_system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": cred_user_prompt_1_ratings}]
        },
    ]
    cred_response, cred_step_outputs_1 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_1,
        tools=tools,
        file_category=["ratings"]
    )
    cred_step_outputs.extend(cred_step_outputs_1)
    # Step 2: from financial bucket
    print("Generating Section 5: Credit Risk... Step 2: Financial bucket")
    initial_input_messages_2 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": cred_system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": cred_user_prompt_2_fin},
                {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 1: \n{cred_response}"}
            ]
        },
    ]
    cred_response, cred_step_outputs_2 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_2,
        tools=tools,  #, tl.gemini_web_search_tool
        file_category=["financial"]
    )
    cred_step_outputs.extend(cred_step_outputs_2)
    # Step 3: from additional bucket
    print("Generating Section 5: Credit Risk... Step 3: Additional bucket")
    initial_input_messages_3 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": cred_system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": cred_user_prompt_3_add},
                {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 2: \n{cred_response}"}
            ]
        },
    ]
    cred_response, cred_step_outputs_3 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_3,
        tools=tools,
        file_category=["additional"]
    )
    cred_step_outputs.extend(cred_step_outputs_3)
    # Step 4: from web search
    print("Generating Section 5: Credit Risk... Step 4: Web search")
    initial_input_messages_4 = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": cred_system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": cred_user_prompt_4_web},
                {"type": "input_text", "text": f"Credit Analysis Markdown Output from Step 3: \n{cred_response}"}
            ]
        },
    ]
    cred_response, cred_step_outputs_4 = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages_4,
        tools=[
            tl.gemini_web_search_tool, tl.get_financial_metrics_yf_api_tool,
            # tl.get_financial_metrics_api_for_chatbot_tool  # tl.get_bonds_api_for_chatbot_tool,
        ],
        file_category=None
    )
    cred_step_outputs.extend(cred_step_outputs_4)
    return re.sub("```.+\n|```\n|```", "", cred_response), cred_step_outputs
    # credit_report_dict.update({"credit_risk": cred_response})
    # reasoning_steps_dict.update({"credit_risk": cred_step_outputs_4})


def generate_executive_summary(company_name, date_and_latest_fin, credit_report_dict):
    ### tool JSON Schema
    # tl = ReasoningToolSchema()
    # load prompts
    #load prompt
    prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
    exec_summary_prompt_file_name = "1_executive_summary.txt"
    with open(prompts_path.joinpath(exec_summary_prompt_file_name), "r") as f:
        exec_summary_prompt = f.read()
    exec_summary_prompt = (exec_summary_prompt.replace("_COMPANY_NAME_", company_name)
                           .replace("_DATE_TODAY_AND_LATEST_REPORTING_PERIOD_", date_and_latest_fin))

    credit_report_md_str_no_summary = (
        credit_report_dict['business_overview'] + '\n\n' + credit_report_dict['financial_health'] + '\n\n'
        + credit_report_dict['debt_structure'] + '\n\n' + credit_report_dict['credit_risk']
    )
    initial_input_messages = [
        {
            "role": "model",
            "content": [{"type": "input_text", "text": exec_summary_prompt}]
        },
        {
            "role": "user",
            "content": [
                # {"type": "input_text", "text": cred_user_prompt_4_web},
                {
                    "type": "input_text",
                    "text": f"Remaining section of the credit report in markdown: \n{credit_report_md_str_no_summary}"
                }
            ]
        },
    ]
    summary_response, summary_step_outputs = gemini_pro_loop_with_tools(
        model_name="o1",
        reasoning_effort="high",
        input_messages=initial_input_messages,
        tools=[tl.rag_retrieval_tool, tl.gemini_web_search_tool],  #, gemini_web_search_tool
        file_category=["financial", "ratings"]
    )
    # credit_report_dict.update({"exec_summary": summary_response})
    # reasoning_steps_dict.update({"exec_summary": summary_step_outputs})
    #
    # credit_report_md = summary_response + "\n\n" + credit_report_md_str_no_summary
    return re.sub("```.+\n|```\n|```", "", summary_response), summary_step_outputs


def generate_credit_report_dummy(company_name, credit_report_dict, reasoning_steps_dict):
    time.sleep(10)
    credit_report_dict = {
        'exec_summary': f"## Executive Summary - {company_name} \n#### This is sample section",
        'business_overview': "## Business Overview \n#### This is sample section",
        'financial_health': "## Financial Health \n#### This is sample section",
        'debt_structure': "## Debt Structure \n#### This is sample section",
        'credit_risk': "## Credit Risk \n#### This is sample section"
    }
    reasoning_steps_dict = {
        'exec_summary': ["## Executive Summary \n#### This is sample reasoning"],
        'business_overview': ["## Business Overview \n#### This is sample reasoning"],
        'financial_health': ["## Financial Health \n#### This is sample reasoning"],
        'debt_structure': ["## Debt Structure \n#### This is sample reasoning"],
        'credit_risk': ["## Credit Risk \n#### This is sample reasoning"]
    }
    return credit_report_dict, reasoning_steps_dict


# generate report use gemini models
def generate_credit_report_gemini(company_name, user_quant_metrics=None):
    credit_report_dict = {
        'exec_summary': None,
        'business_overview': None,
        'financial_health': None,
        'debt_structure': None,
        'credit_risk': None
    }

    # load files classifying from gcs blob
    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

    blob = bucket.blob(f'generated/{company_name}/class_file_dict.json')
    contents = blob.download_as_bytes()
    class_file_dict = json.loads(contents.decode("utf-8"))

    # initialise vertexai
    vertexai.init(credentials=gcloud_connect_credentials(), project=os.getenv('GCLOUD_PROJECT_ID'), location="us-central1",)

    file_blobs = storage_client.list_blobs(os.getenv('GCLOUD_BUCKET_NAME'), prefix='companies/' + company_name + '/')
    file_list = [blob for blob in file_blobs if blob.name != 'companies/' and blob.name != 'companies/' + company_name + '/']
    # load file uri from class_file_dict
    financial_file_blob_list, bond_file_blob_list, ratings_file_blob_list, additional_file_blob_list = [], [], [], []
    for blob in file_list:
        gsutil_uri = 'gs://' + f"{os.getenv('GCLOUD_BUCKET_NAME')}/" + f'{blob.name}'
        if gsutil_uri in class_file_dict['financial']:
            financial_file_blob_list.append(blob)
        elif gsutil_uri in class_file_dict['bond']:
            bond_file_blob_list.append(blob)
        elif gsutil_uri in class_file_dict['ratings']:
            ratings_file_blob_list.append(blob)
        else:
            additional_file_blob_list.append(blob)

    ####### section: business model overview
    print("Generating Section 2: Business Overview...")
    #load prompt
    PROMPTS_PATH = pathlib.Path().parent.joinpath("./prompts").resolve()
    business_overview_prompt_file_name = "2_business_overview.txt"
    with open(PROMPTS_PATH.joinpath(business_overview_prompt_file_name), "r") as f:
        business_overview_prompt = f.read()
    business_overview_prompt = business_overview_prompt.replace("_COMPANY_NAME_", company_name)

    # need to upload files when using genai
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    financial_files_content_parts_genai = [
        client.files.upload(
            file=io.BytesIO(blob.download_as_bytes()),
            config=dict(mime_type=blob.content_type)
        ) for blob in financial_file_blob_list
    ]
    contents = [
        types.Content(
            role='user',
            parts=[
                types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
                for file in financial_files_content_parts_genai
            ]
        ),
        types.Content(role='user', parts=[types.Part.from_text(text=business_overview_prompt)])
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="text/plain"
    )
    response = client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05", #"gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    )
    # list all the search grounding
    href_markdown = "\nRefs:\n\n"
    for i, chunk in enumerate(response.candidates[0].grounding_metadata.grounding_chunks, 1):
        href_markdown += f"[{chunk.web.title}]({chunk.web.uri})\n\n" # {i}.

    response_str = re.sub("```.+\n|```\n|```", "", response.text + href_markdown)
    credit_report_dict['business_overview'] = response_str

    ####### section: financial health
    print("Generating Section 3: Financial Health...")
    # load prompts
    # MAIN_PATH = pathlib.Path(__file__).parent
    # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
    PROMPTS_PATH = pathlib.Path().parent.joinpath("./prompts").resolve()
    fin_system_prompt_file_name = "3_financial_health_system_prompt.txt"
    fin_user_prompt_1_metric_file_name = "3_financial_health_user_prompt_1_metric.txt"
    fin_user_prompt_1_user_metric_file_name = "3_financial_health_user_prompt_1_user_quant_table.txt"
    fin_user_prompt_2_analyse_file_name = "3_financial_health_user_prompt_2_analyse.txt"

    with open(PROMPTS_PATH.joinpath(fin_system_prompt_file_name), "r") as f:
        fin_system_prompt = f.read()

    fin_system_prompt = fin_system_prompt.replace("_COMPANY_NAME_", company_name)

    with open(PROMPTS_PATH.joinpath(fin_user_prompt_1_metric_file_name), "r") as f:
        fin_user_prompt_1_metric = f.read()

    with open(PROMPTS_PATH.joinpath(fin_user_prompt_1_user_metric_file_name), "r") as f:
        fin_user_prompt_1_user_metric = f.read()

    with open(PROMPTS_PATH.joinpath(fin_user_prompt_2_analyse_file_name), "r") as f:
        fin_user_prompt_2_analyse = f.read()

    if user_quant_metrics is not None:
        fin_user_prompt_1_metric = fin_user_prompt_1_metric.replace(
            "_USER_QUANT_METRICS_", "For the user provided quantitative metrics:\n" + user_quant_metrics
        )
        fin_user_prompt_1_metric = fin_user_prompt_1_metric.replace(
            "_USER_QUANT_METRIC_TABLE_", fin_user_prompt_1_user_metric
        )
    else:
        fin_user_prompt_1_metric = fin_user_prompt_1_metric.replace("_USER_QUANT_METRICS_", "")
        fin_user_prompt_1_metric = fin_user_prompt_1_metric.replace("_USER_QUANT_METRIC_TABLE_", "")

    # Step 1: create metric table

    # client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # contents = [
    #     types.Content(
    #         role='user',
    #         parts=[
    #             types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
    #             for file in financial_files_content_parts_genai
    #         ]
    #     ),
    #     types.Content(role='user', parts=[types.Part.from_text(text=fin_user_prompt_1_metric)])
    # ]
    # generate_content_config = types.GenerateContentConfig(
    #     temperature=1,
    #     top_p=0.95,
    #     top_k=64,
    #     max_output_tokens=8192,
    #     # tools=[types.Tool(google_search=types.GoogleSearch())],
    #     response_mime_type="text/plain",
    #     system_instruction=[types.Part.from_text(text=fin_system_prompt)]
    # )
    # fin_metric_response = client.models.generate_content(
    #     model="gemini-2.0-flash", #"gemini-2.0-flash",
    #     contents=contents,
    #     config=generate_content_config,
    # )

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8190,
        "response_mime_type": "text/plain",
    }

    model = GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        system_instruction=fin_system_prompt,
        generation_config=generation_config,
    ) # gemini-2.0-flash-thinking-exp-01-21 gemini-2.0-flash-001 gemini-2.0-pro-exp-02-05

    # load and process files
    financial_files_content_parts_vertexai = []
    for file_blob in financial_file_blob_list:
        files_content_parts = process_files_content_parts(os.getenv('GCLOUD_BUCKET_NAME'), file_blob, vertexai=True)
        financial_files_content_parts_vertexai.extend(files_content_parts)

    fin_metric_response = model.generate_content(
        contents=[
            Content(
                role="user",
                parts=financial_files_content_parts_vertexai + [Part.from_text(fin_user_prompt_1_metric)]
            ),
        ]
    )
    fin_metric_response_str = re.sub("```.+\n|```\n|```", "", fin_metric_response.text)

    # Step 2: write financial health anaylse
    fin_analysis_response = model.generate_content(
        contents=[
            Content(
                role="user",
                parts=financial_files_content_parts_vertexai + [Part.from_text(fin_metric_response_str), Part.from_text(fin_user_prompt_2_analyse)]
            ),
        ]
    )
    fin_response_str = fin_metric_response_str + '\n\n' + re.sub("```.+\n|```\n|```", "", fin_analysis_response.text)
    credit_report_dict['financial_health'] = fin_response_str

    # # get financial metrics from Alpha Vantage API
    # income_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={company_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    # r = requests.get(income_url)
    # income_data = r.json()
    #
    # bal_url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={company_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    # r = requests.get(bal_url)
    # bal_data = r.json()
    #
    # cash_url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={company_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    # r = requests.get(cash_url)
    # cash_data = r.json()


    ####### section: debt structure
    print("Generating Section 4: Debt Structure and Maturity Profile...")
    # load prompts
    # MAIN_PATH = pathlib.Path(__file__).parent
    # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
    PROMPTS_PATH = pathlib.Path().parent.joinpath("./prompts").resolve()
    system_prompt_file_name = "4_debt_structure_system_prompt.txt"
    user_prompt_1_web_file_name = "4_debt_structure_user_prompt_1_web.txt"
    # user_prompt_2_list_file_name = "4_debt_structure_user_prompt_2_list.txt"
    user_prompt_2_bond_file_name = "4_debt_structure_user_prompt_2_bond.txt"
    with open(PROMPTS_PATH.joinpath(system_prompt_file_name), "r") as f:
        system_prompt = f.read()

    debt_struc_system_prompt = system_prompt.replace("_COMPANY_NAME_", company_name)

    with open(PROMPTS_PATH.joinpath(user_prompt_1_web_file_name), "r") as f:
        debt_struc_user_prompt_1_web = f.read()

    with open(PROMPTS_PATH.joinpath(user_prompt_2_bond_file_name), "r") as f:
        debt_struc_user_prompt_2_bond = f.read()

    # 1st user prompt: list web search bonds + docs
    # need to upload files when using genai
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    bond_files_content_parts_genai = [
        client.files.upload(
            file=io.BytesIO(blob.download_as_bytes()),
            config=dict(mime_type=blob.content_type)
        ) for blob in bond_file_blob_list
    ]
    contents = [
        types.Content(
            role='user',
            parts=[
                types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
                for file in bond_files_content_parts_genai
            ]
        ),
        types.Content(role='user', parts=[types.Part.from_text(text=debt_struc_user_prompt_1_web)])
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=debt_struc_system_prompt)]
    )
    response_1_web = client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05", #"gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    )
    # list all the search grounding
    href_markdown = "Refs:\n\n"
    for i, chunk in enumerate(response_1_web.candidates[0].grounding_metadata.grounding_chunks, 1):
        href_markdown += f"{i}. [{chunk.web.title}]({chunk.web.uri})\n\n"

    response_bond_list_str = re.sub("```.+\n|```\n|```", "", response_1_web.text + href_markdown)

    # step 2: construct markdown
    # load and process files
    bond_files_content_parts_vertexai = []
    for file_blob in bond_file_blob_list:
        files_content_parts = process_files_content_parts(os.getenv('GCLOUD_BUCKET_NAME'), file_blob, vertexai=True)
        bond_files_content_parts_vertexai.extend(files_content_parts)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 65535,
        "response_mime_type": "text/plain",
    }

    model = GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        system_instruction=debt_struc_system_prompt,
        generation_config=generation_config,
    ) # "gemini-2.0-flash-thinking-exp-01-21" gemini-2.0-flash-001 gemini-2.0-pro-exp-02-05

    bond_files_contents_prompt_1 = [
        Content(
            role="user",
            parts=bond_files_content_parts_vertexai
                  + [Part.from_text(response_bond_list_str), Part.from_text(debt_struc_user_prompt_2_bond)]
        ),
    ]
    responses = model.generate_content(contents=bond_files_contents_prompt_1)

    bond_struc_resp_1_str = re.sub("```.+\n|```\n|```", "", responses.text)
    credit_report_dict['debt_structure'] = bond_struc_resp_1_str

    ####### section: credit risk
    print("Generating Section 5: Credit Risk...")
    # MAIN_PATH = pathlib.Path(__file__).parent
    # PROMPTS_PATH = MAIN_PATH.joinpath("../prompts").resolve()
    PROMPTS_PATH = pathlib.Path().parent.joinpath("./prompts").resolve()
    cred_system_prompt_file_name = "5_credit_risk_system_prompt.txt"
    cred_user_prompt_1_ratings_file_name = "5_credit_risk_user_prompt_1_ratings.txt"
    cred_user_prompt_2_fin_file_name = "5_credit_risk_user_prompt_2_financial.txt"
    cred_user_prompt_3_add_file_name = "5_credit_risk_user_prompt_3_additional.txt"
    cred_user_prompt_4_web_file_name = "5_credit_risk_user_prompt_4_web.txt"
    with open(PROMPTS_PATH.joinpath(cred_system_prompt_file_name), "r") as f:
        cred_system_prompt = f.read()

    cred_system_prompt = cred_system_prompt.replace("_COMPANY_NAME_", company_name)

    with open(PROMPTS_PATH.joinpath(cred_user_prompt_1_ratings_file_name), "r") as f:
        cred_user_prompt_1_ratings = f.read()

    with open(PROMPTS_PATH.joinpath(cred_user_prompt_2_fin_file_name), "r") as f:
        cred_user_prompt_2_fin = f.read()

    with open(PROMPTS_PATH.joinpath(cred_user_prompt_3_add_file_name), "r") as f:
        cred_user_prompt_3_add = f.read()

    with open(PROMPTS_PATH.joinpath(cred_user_prompt_4_web_file_name), "r") as f:
        cred_user_prompt_4_web = f.read()

    # Step 1: rating reports

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 65535,
        "response_mime_type": "text/plain",
    }

    model = GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        system_instruction=cred_system_prompt,
        generation_config=generation_config,
    ) # "gemini-2.0-flash-thinking-exp-01-21" gemini-2.0-flash-001 gemini-2.0-pro-exp-02-05

    # load and process files
    ratings_files_content_parts_vertexai = []
    for file_blob in ratings_file_blob_list:
        files_content_parts = process_files_content_parts(os.getenv('GCLOUD_BUCKET_NAME'), file_blob, vertexai=True)
        ratings_files_content_parts_vertexai.extend(files_content_parts)

    cred_response = model.generate_content(
        contents=[
            Content(
                role="user",
                parts=ratings_files_content_parts_vertexai + [Part.from_text(cred_user_prompt_1_ratings)]
            ),
        ]
    )
    cred_response_str = re.sub("```.+\n|```\n|```", "", cred_response.text)

    # Step 2: look at financial docs
    # load and process files
    financial_files_content_parts_vertexai = []
    for file_blob in financial_file_blob_list:
        files_content_parts = process_files_content_parts(os.getenv('GCLOUD_BUCKET_NAME'), file_blob, vertexai=True)
        financial_files_content_parts_vertexai.extend(files_content_parts)

    cred_response = model.generate_content(
        contents=[
            Content(
                role="user",
                parts=financial_files_content_parts_vertexai
                      + [Part.from_text(cred_response_str), Part.from_text(cred_user_prompt_2_fin)]
            ),
        ]
    )
    cred_response_str = re.sub("```.+\n|```\n|```", "", cred_response.text)

    # Step 3: look at additional docs
    # load and process files
    additional_files_content_parts_vertexai = []
    for file_blob in additional_file_blob_list:
        files_content_parts = process_files_content_parts(os.getenv('GCLOUD_BUCKET_NAME'), file_blob, vertexai=True)
        additional_files_content_parts_vertexai.extend(files_content_parts)

    cred_response = model.generate_content(
        contents=[
            Content(
                role="user",
                parts=additional_files_content_parts_vertexai
                      + [Part.from_text(cred_response_str), Part.from_text(cred_user_prompt_3_add)]
            ),
        ]
    )
    cred_response_str = re.sub("```.+\n|```\n|```", "", cred_response.text)

    # Step 4: web search
    # need to upload files when using genai
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    contents = [
        types.Content(role='user', parts=[types.Part.from_text(text=cred_response_str)]),
        types.Content(role='user', parts=[types.Part.from_text(text=cred_user_prompt_4_web)])
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=cred_system_prompt)]
    )
    cred_response_web = client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05", #"gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    )
    # list all the search grounding
    href_markdown = "Refs:\n\n"
    if cred_response_web.candidates[0].grounding_metadata.grounding_chunks is not None:
        for i, chunk in enumerate(cred_response_web.candidates[0].grounding_metadata.grounding_chunks, 1):
            href_markdown += f"{i}. [{chunk.web.title}]({chunk.web.uri})\n\n"
    else:
        href_markdown = ""

    cred_response_web_str = re.sub("```.+\n|```\n|```", "", cred_response_web.text + href_markdown)

    # save final section output to dictionary
    credit_report_dict['credit_risk'] = cred_response_web_str


    ####### section: executive summary
    print("Generating Section 1: Executive Summary...")
    # get today's date
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    contents = [
        types.Content(
            role='user',
            parts=[types.Part.from_text(text="What is today's date? Respond only the date without any other words or comments.")]
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="text/plain"
    )
    date_response = client.models.generate_content(
        model="gemini-2.0-flash", #"gemini-2.0-flash", gemini-2.0-pro-exp-02-05
        contents=contents,
        config=generate_content_config,
    )
    date_response_str = re.sub("```.+\n|```\n|```", "", date_response.text)

    # load prompts
    #load prompt
    PROMPTS_PATH = pathlib.Path().parent.joinpath("./prompts").resolve()
    exec_summary_prompt_file_name = "1_executive_summary.txt"
    with open(PROMPTS_PATH.joinpath(exec_summary_prompt_file_name), "r") as f:
        exec_summary_prompt = f.read()
    exec_summary_prompt = exec_summary_prompt.replace("_COMPANY_NAME_", company_name)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 65535,
        "response_mime_type": "text/plain",
    }

    model = GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        # system_instruction=cred_system_prompt,
        generation_config=generation_config,
    ) # "gemini-2.0-flash-thinking-exp-01-21" gemini-2.0-flash-001 gemini-2.0-pro-exp-02-05

    # load and process files
    # ratings_files_content_parts_vertexai = []
    # for file_blob in ratings_file_blob_list:
    #     files_content_parts = process_files_content_parts(bucket_name, file_blob, vertexai=True)
    #     ratings_files_content_parts_vertexai.extend(files_content_parts)

    # combine all dict entries into single markdown string
    credit_report_md_str_no_summary = (
            credit_report_dict['business_overview'] + '\n\n' + credit_report_dict['financial_health'] + '\n\n'
            + credit_report_dict['debt_structure'] + '\n\n' + credit_report_dict['credit_risk']
    )
    exec_summary_prompt = exec_summary_prompt.replace("_DATE_OF_REPORT_", date_response_str)

    exec_response = model.generate_content(
        contents=[
            Content(
                role="user",
                parts=ratings_files_content_parts_vertexai
                      + [Part.from_text(credit_report_md_str_no_summary), Part.from_text(exec_summary_prompt)]
            ),
        ]
    )
    exec_response_str = re.sub("```.+\n|```\n|```", "", exec_response.text)
    credit_report_dict['exec_summary'] = exec_response_str

    # save full credit report as markdown
    print("Saving credit report as markdown.")
    credit_report_md_str = exec_response_str + '\n\n' + credit_report_md_str_no_summary

    with open(f'./data/{company_name}_credit_report.md', 'w') as f:
        f.write(credit_report_md_str)


