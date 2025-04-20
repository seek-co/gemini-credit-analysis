import io
import time
import os
import glob
import json
import extract_msg
import pandas as pd

import vertexai.generative_models as gm
import weaviate
import weaviate.classes.query as wq
import copy
import PIL
import tiktoken
import yfinance as yf
import requests

from tqdm import tqdm
from io import BytesIO
from docx import Document
from base64 import b64decode
from weaviate.classes.tenants import Tenant
from weaviate.util import generate_uuid5
from weaviate.classes.init import Auth
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.msg import partition_msg
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.csv import partition_csv
from unstructured.partition.image import partition_image
from unstructured.cleaners.core import group_broken_paragraphs
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Image, Table
# (
#     Text, Formula, FigureCaption, NarrativeText, ListItem, Title, Address, EmailAddress, , PageBreak, ,
#     Header, Footer, CodeSnippet, PageNumber
# )

from google import genai
from google.genai import types, client
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import storage_control_v2



# load .env if not in production
if not bool(os.getenv("PRODUCTION_MODE")):
    from dotenv import load_dotenv
    load_dotenv()
    
print("From env:", os.getenv("GEMINI_API_KEY"))
# Check what's actually in the .env file
import dotenv
env_values = dotenv.dotenv_values()
print("From .env file:", env_values.get("GEMINI_API_KEY"))

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def extract_image_data_with_description(element, file_path, file_category, image_explainer_prompt):

    page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
    # image_path = element.metadata.image_path if hasattr(element.metadata, 'image_path') else None
    image_base64 = element.metadata.image_base64 if hasattr(element.metadata, 'image_base64') else None
    image_mime_type = element.metadata.image_mime_type if hasattr(element.metadata, 'image_mime_type') else None

    if image_base64 is not None:
        response = client_gemini.models.generate_content(
            model="gemini-1.5-pro-latest",
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(text=f"{image_explainer_prompt}"),
                    types.Part.from_bytes(data=image_base64, mime_type=image_mime_type)
                ])
            ]
        )
        description = response.text
    else:
        description = "Empty image"

    image_data = [{
        "source_document": file_path,
        "file_category": file_category,
        "page_number": page_number,
        # "image_path": image_path,
        "mime_type": image_mime_type,
        "base64_encoding": image_base64,
        "description": description,
        "content_type": "image"
    }]
    return image_data


def extract_table_data_with_description(element, file_path, file_category, table_explainer_prompt):
    page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
    # table content in text string
    table_text = str(element)
    # table content in html string
    table_html = element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else None
    table_base64 = element.metadata.image_base64 if hasattr(element.metadata, 'image_base64') else None
    image_mime_type = element.metadata.image_mime_type if hasattr(element.metadata, 'image_mime_type') else None

    # Generate summary using the model
    content_parts = [types.Part.from_text(text=f"{table_explainer_prompt} \nTable in HTML: \n{table_html}")]
    if table_base64 is not None:
        content_parts.append(types.Part(inline_data=types.Blob(mime_type=image_mime_type, data=b64decode(table_base64))))
    else:
        content_parts.append(types.Part.from_text(text=f"No image found for table"))

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[types.Content(role="user", parts=content_parts)],
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    description = response.text

    table_data = [{
        "source_document": file_path,
        "file_category": file_category,
        "page_number": page_number,
        "table_text": table_text,
        "table_html": table_html,
        "base64_encoding": table_base64,
        "mime_type": image_mime_type,
        "description": description,
        "content_type": "table"
    }]
    return table_data


def extract_narrative_text_data(element, file_path, file_category, part_counters):

    page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else None

    if page_number not in part_counters:
        part_counters[page_number] = 1
    else:
        part_counters[page_number] += 1

    part_number = part_counters[page_number]

    text_content = element.text
    # check token numbers, chunk if too long
    num_token = num_tokens_from_string(text_content, "o200k_base")
    if num_token > 8100:
        chunks = chunk_elements(elements=[element], max_characters=10000, overlap=500, include_orig_elements=True)
        # extract for each chunk
        text_data = [{
            "source_document": file_path,
            "file_category": file_category,
            "page_number": page_number,
            "part_number": part_number,
            "text": ch.text,
            "content_type": "text"
        } for ch in chunks]
    else:
        text_data = [{
            "source_document": file_path,
            "file_category": file_category,
            "page_number": page_number,
            "part_number": part_number,
            "text": text_content,
            "content_type": "text"
        }]
    return text_data


def extract_other_text_data(element, file_path, file_category):
    page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
    text_content = element.text
    text_data = [{
        "source_document": file_path,
        "file_category": file_category,
        "page_number": page_number,
        "text": text_content,
        "content_type": "text"
    }]
    return text_data


def extract_chunks_text(chunks, file_path, file_category):
    text_data = []
    for chunk in chunks:
        # combine all the page numbers that exist in the original elements under this chunk
        page_chunk = ""
        last_pg_num = 0
        for elem in chunk.metadata.orig_elements:
            new_pg_num = elem.metadata.page_number
            if hasattr(elem.metadata, 'page_number') and new_pg_num != last_pg_num:
                page_chunk += str(new_pg_num) + "/"
                last_pg_num = new_pg_num
            else:
                page_chunk += ""

        text_data.append({
            "source_document": file_path,
            "file_category": file_category,
            "page_number": None,
            "page_chunk": page_chunk,
            "text": chunk.text,
            "content_type": "text"
        })
    return text_data


# extract all image, table, text data into lists
def extract_data_list(file_elements, file_path, file_category): # , image_explainer_prompt=None, table_explainer_prompt=None
    # table_explainer_prompt = (
    #     "You are an expert financial analyst that is very experienced in analysing financial credit conditions "
    #     "of companies. You will be provided with table contents in HTML. "
    #     "Your task is to explain the table contents in detail. "
    #     "Deliver a coherent, factual explanation for every row and column of the table, and all of the table entries. "
    #     "Limit your response output to a maximum of 6500 words, or anything less depending on the explanation needs. "
    #     "It is VERY IMPORTANT NOT to exceed the maximum of 6500 words limit for your response. "
    #     "Otherwise your response will receive a very serious penalty later!"
    # )
    table_explainer_prompt = (
        "You are an expert financial data analyst that is very experienced in analysing financial credit conditions "
        "of companies. You will be provided with table contents in HTML. Your task is to explain the "
        "table content in detail, including the the actual metrics, units, entry values and the true meaning of "
        "each column and each row, according to their row and column names that you identify. "
        "Avoid bullet points, instead, you must deliver a coherent, factual explanation for every row and column "
        "of the table, and all of the table entries. Your response will be vectorised and stored in a vector database "
        "for RAG retrieval later. It is extremely important to turn the provided table content into a text explanation "
        "that is easy for information retrieval later. Limit your response output to a maximum of 6500 words, "
        "or anything less depending on the explanation needs. It is VERY IMPORTANT NOT to exceed the maximum limit "
        "for your response. Otherwise your response will receive a very serious penalty later!"
    )
    image_explainer_prompt = (
        "You are an expert financial analyst that is very experienced in analysing financial credit conditions "
        "of companies. You will be provided with images that are found within company financial documents. "
        "Identify whether the image is informative and relevant to company financials, "
        "or the image is for artistic purposes only. If the image is informative and relevant to company financials, "
        "describe the type of visual (e.g., chart, photograph, infographic) and its key elements. "
        "If the image is for artistic purposes only, describe the visual type as artistic, "
        "and describe briefly what is shown in the image. "
        "Highlight significant data points or trends that are relevant to financial analysis. "
        "Limit your response output to a maximum of 6500 words, or anything less depending on the explanation needs. "
        "It is VERY IMPORTANT NOT to exceed the maximum of 6500 words limit for your response. "
        "Otherwise your response will receive a very serious penalty later!"
    )
    image_data, table_data, text_data = [], [], [] #narrative_text_data, other_text_data = [], [], [], []
    # part_counters = {}
    try:
        for i, element in enumerate(file_elements, 1):
            if isinstance(element, Image):
                extracted_image_data = extract_image_data_with_description(
                    element=element, file_path=file_path, file_category=file_category,
                    image_explainer_prompt=image_explainer_prompt
                )
                for extracted_img in extracted_image_data:
                    if extracted_img["base64_encoding"] is not None:
                        image_data.append(extracted_img)
            elif isinstance(element, Table):
                table_data.extend(
                    extract_table_data_with_description(
                        element=element, file_path=file_path, file_category=file_category,
                        table_explainer_prompt=table_explainer_prompt
                    )
                )

        chunks = chunk_elements(elements=file_elements, max_characters=1000, overlap=500, include_orig_elements=True)
        text_data = extract_chunks_text(chunks=chunks, file_path=file_path, file_category=file_category)
        return image_data, table_data, text_data #narrative_text_data, other_text_data
    except Exception as exc:
        raise f"{exc}"


def get_embedding(text):
    response = client_gemini.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])]
    )
    embd = response.embeddings[0].values
    return embd


# ingest data list into weaviate
def ingest_text_data(tenant, text_data):
    with tenant.batch.dynamic() as batch:
        for i, text_obj in enumerate(tqdm(text_data, desc="Ingesting chunks text data"), 1):
            vector = get_embedding(text_obj['text'])
            batch.add_object(
                properties=text_obj,
                uuid=generate_uuid5(
                    f"{text_obj['source_document']}_{text_obj['page_number']}_{text_obj['text'][:200].replace(' ', '')}"),
                vector=vector
            )
            if i % 150 == 0:
                time.sleep(60)


# ingest data list into weaviate
def ingest_narrative_text_data(tenant, text_data):
    with tenant.batch.dynamic() as batch:
        for i, text_obj in enumerate(tqdm(text_data, desc="Ingesting narrative text data"), 1):
            vector = get_embedding(text_obj['text'])
            batch.add_object(
                properties=text_obj,
                uuid=generate_uuid5(
                    f"{text_obj['source_document']}_{text_obj['page_number']}_{text_obj['part_number']}_{text_obj['text'][:200].replace(' ', '')}"),
                vector=vector
            )
            if i % 150 == 0:
                time.sleep(60)


def ingest_other_text_data(tenant, text_data):
    with tenant.batch.dynamic() as batch:
        for i, text_obj in enumerate(tqdm(text_data, desc="Ingesting other text data"), 1):
            vector = get_embedding(text_obj['text'])
            batch.add_object(
                properties=text_obj,
                uuid=generate_uuid5(
                    f"{text_obj['source_document']}_{text_obj['page_number']}_{text_obj['text'][:200].replace(' ', '')}"),
                vector=vector
            )
            if i % 150 == 0:
                time.sleep(60)


def ingest_image_data(tenant, image_data):
    with tenant.batch.dynamic() as batch:
        for i, image_obj in enumerate(tqdm(image_data, desc="Ingesting image data"), 1):
            vector = get_embedding(image_obj['description'])
            batch.add_object(
                properties=image_obj,
                uuid=generate_uuid5(
                    f"{image_obj['source_document']}_{image_obj['page_number']}_{image_obj['description'][:100].replace(' ', '')}"),
                vector=vector
            )
            if i % 150 == 0:
                time.sleep(60)


def ingest_table_data(tenant, table_data):
    with tenant.batch.dynamic() as batch:
        for i, table_obj in enumerate(tqdm(table_data, desc="Ingesting table data"), 1):
            vector = get_embedding(table_obj['description'])
            batch.add_object(
                properties=table_obj,
                uuid=generate_uuid5(
                    f"{table_obj['source_document']}_{table_obj['page_number']}_{table_obj['table_text'][:100].replace(' ', '')}"),
                vector=vector
            )
            if i % 150 == 0:
                time.sleep(60)


def ingest_all_data(tenant_name, image_data, table_data, text_data, collection="Companies"):
    client_weaviate = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    multi_collection_companies = client_weaviate.collections.get(collection)
    tenant = multi_collection_companies.with_tenant(tenant_name)

    ingest_text_data(tenant, text_data)
    ingest_image_data(tenant, image_data)
    ingest_table_data(tenant, table_data)
    if len(tenant.batch.failed_objects) > 0:
        print(f"Failed to import {len(tenant.batch.failed_objects)} objects")
    else:
        print("All objects imported successfully")

    client_weaviate.close()


def partition_file_to_elements(file_path, file_content=None):
    # if file_content is provided, then file_path is just file name.
    # only pass io Bytes content to partition functions
    partition_kwargs = {"file": io.BytesIO(b64decode(file_content))} if file_content is not None else {"filename": file_path}
    if os.path.normpath(file_path).endswith(".msg"):
        # handle .msg
        if file_content is not None:
            file_elements = partition_msg(file=io.BytesIO(b64decode(file_content)), process_attachments=False)
            # msg_parser is not too good at handling attachments, do it manually.
        else:
            file_elements = partition_msg(
                filename=file_path,
                process_attachments=False
            )
            # msg_parser is not too good at handling attachments, do it manually.
        msg = extract_msg.Message(b64decode(file_content))  # file_path_ls[4] temp_dl_path
        attachment_ls = [attachment for attachment in msg.attachments]

        # handle pdf or img or xlsx or docx attachments only
        for attachment in attachment_ls:
            if "image" in attachment.mimetype:
                att_elements = partition_image(
                    file=io.BytesIO(attachment.data),
                    infer_table_structure=True,
                    extract_images_in_pdf=True,  # mandatory to set as ``True``
                    extract_image_block_types=["Image", "Table"],  # optional ,
                    extract_image_block_to_payload=True,  # optional
                    strategy="ocr_only"
                )
            elif attachment.mimetype == "application/pdf":
                att_elements = partition_pdf(
                    file=io.BytesIO(attachment.data),
                    strategy='hi_res',
                    infer_table_structure=True,
                    extract_images_in_pdf=True,  # mandatory to set as ``True``
                    extract_image_block_types=["Image", "Table"],  # optional
                    extract_image_block_to_payload=True,  # optional
                )
            elif attachment.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # handle .docx
                att_elements = partition_docx(
                    file=io.BytesIO(attachment.data),
                    infer_table_structure=True,
                    include_page_breaks=True,
                )
            elif attachment.mimetype == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                # handle .xlsx
                att_elements = partition_xlsx(
                    file=io.BytesIO(attachment.data),
                    infer_table_structure=True,
                    find_subtable=True,
                )
            file_elements.extend(att_elements)
    elif os.path.normpath(file_path).endswith(".docx"):
        # handle .docx
        file_elements = partition_docx(
            **partition_kwargs,
            infer_table_structure=True,
        )
    elif os.path.normpath(file_path).endswith(".txt"):
        # handle .txt
        with open(file_path, "r") as f:
            text = f.read()

        file_elements = partition_text(
            text=text,
            max_partition=1000,
            paragraph_grouper=group_broken_paragraphs
        )
    elif os.path.normpath(file_path).endswith(".csv"):
        # handle .csv
        file_elements = partition_csv(
            **partition_kwargs,
            infer_table_structure=True,
            find_subtable=False  #True
        )
    elif os.path.normpath(file_path).endswith(".xlsx"):
        # handle .xlsx
        file_elements = partition_xlsx(
            **partition_kwargs,
            infer_table_structure=True,
            find_subtable=True,
        )
    elif os.path.normpath(file_path).endswith(".pdf"):
        file_elements = partition_pdf(
            **partition_kwargs,
            strategy='hi_res',
            infer_table_structure=True,
            extract_images_in_pdf=True,  # mandatory to set as ``True``
            extract_image_block_types=["Image", "Table"],  # optional
            extract_image_block_to_payload=True,  # optional
        )
    elif os.path.normpath(file_path).endswith(".png") or os.path.normpath(file_path).endswith(".jpg"):
        # handle images
        # convert png to rgb first o/w some rgba png will fail
        im = PIL.Image.open(io.BytesIO(b64decode(file_content)) if file_content is not None else file_path)
        rgb_im = im.convert("RGB")
        if file_content is not None:
            im_byte_arr = io.BytesIO()
            im.save(im_byte_arr, format=im.format)
            partition_kwargs.update({"file_content": im_byte_arr})
        else:
            rgb_im.save(file_path)

        file_elements = partition_image(
            **partition_kwargs,
            infer_table_structure=True,
            extract_images_in_pdf=True,  # mandatory to set as ``True``
            extract_image_block_types=["Image", "Table"],  # optional ,
            extract_image_block_to_payload=True,  # optional
            strategy="hi_res"
        )

    return file_elements


def parse_new_company_file_to_vector_db(file_path, company_name, file_category, file_content=None):
    company_name_vector_db = change_company_name_gsc_to_vectordb(company_name=company_name) #company_name.replace(" ", "_").replace("(", "_").replace(")", "_")
    file_elements = partition_file_to_elements(file_path=file_path, file_content=file_content)
    image_data, table_data, text_data = extract_data_list(
        file_elements=file_elements, file_path=file_path, file_category=file_category
    )
    ingest_all_data(
        tenant_name=company_name_vector_db, image_data=image_data, table_data=table_data, text_data=text_data
        # narrative_text_data=narrative_text_data, other_text_data=other_text_data
    )


def parse_new_company_folder_to_vector_db(company_folder_path, company_name): # , image_explainer_prompt, table_explainer_prompt
    company_name_vector_db = change_company_name_gsc_to_vectordb(company_name=company_name) #company_name.replace(" ", "_").replace("(", "_").replace(")", "_")

    # create new tenant for the company
    client_weaviate = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    multi_collection_companies = client_weaviate.collections.get("Companies")
    multi_collection_companies.tenants.create(
        tenants=[Tenant(name=company_name_vector_db), ]
    )
    client_weaviate.close()

    # load file categories from gcs blob
    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    bucket_name = os.getenv('GCLOUD_BUCKET_NAME')
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(f'generated/{company_name}/class_file_dict.json')
    contents = blob.download_as_bytes()
    class_file_dict = json.loads(contents.decode("utf-8"))

    storage_client.close()

    # convert uri path to file names
    class_file_name_dict = {}
    for category, file_path_list in class_file_dict.items():
        file_name_ls = []
        for file_path in file_path_list:
            sep_file_path_ls = file_path.split('/', maxsplit=100)
            file_name = sep_file_path_ls[-2] if sep_file_path_ls[-1] == "" else sep_file_path_ls[-1]
            file_name_ls.append(file_name)
        class_file_name_dict.update({category: file_name_ls})

    # list all files in company folder
    recursive_file_path_ls = glob.glob(os.path.abspath(company_folder_path) + '**/**', recursive=True)
    file_path_ls = [
        file_path for file_path in recursive_file_path_ls if
        not os.path.isdir(file_path) and not os.path.basename(os.path.normpath(file_path)).startswith(".")
    ]
    for i, file_path in enumerate(file_path_ls, 1):
        # identify the file category
        file_name_from_path = os.path.basename(os.path.normpath(file_path))

        print(f"Ingesting file {i} of {len(file_path_ls)}: {file_name_from_path}")

        file_category = "financial" if file_name_from_path in class_file_name_dict['financial'] \
            else "bond" if file_name_from_path in class_file_name_dict["bond"] \
            else "ratings" if file_name_from_path in class_file_name_dict["ratings"] \
            else "additional" if file_name_from_path in class_file_name_dict["additional"] \
            else None

        if file_category is None:
            raise "file_category is None, please check code."

        file_elements = partition_file_to_elements(file_path=file_path)
        image_data, table_data, text_data = extract_data_list(
            file_elements=file_elements, file_path=file_path, file_category=file_category,
        )
        ingest_all_data(
            tenant_name=company_name_vector_db, image_data=image_data, table_data=table_data, text_data=text_data
        )



####### vec db query, llm tools
def search_multimodal(company_tenant, query: str, file_category: list[str] = None, alpha: float = 0.75, limit: int = 20):
    query_vector = get_embedding(query)
    if file_category is not None:
        response = company_tenant.query.hybrid(
            query=query, vector=query_vector, alpha=alpha, limit=limit,
            filters=wq.Filter.any_of([
                wq.Filter.by_property("file_category").equal(category) for category in file_category
            ]),
            return_metadata=wq.MetadataQuery(score=True, explain_score=True),
        )
    else:
        response = company_tenant.query.hybrid(
            query=query, vector=query_vector, alpha=alpha, limit=limit,
            return_metadata=wq.MetadataQuery(score=True, explain_score=True),
        )
    return response.objects


def parse_search_response_objects_old(search_response_objects):
    # parse text and tables into strings, and for images get base64 to be attached to llm in prompt
    response_object_json_ls = []
    retrieval_str = ""
    for i, item in enumerate(search_response_objects, 1):
        object_properties_dict = copy.deepcopy(item.properties)
        object_properties_dict.update({
            "score": item.metadata.score, "explain_score": item.metadata.explain_score
        })
        response_object_json_ls.append(object_properties_dict)

        # retrieval string text that can be passed directly to llm
        retrieval_str += f"RAG Context Item Number: {i}, Type: {item.properties['content_type']}, File Category: \n"
        if item.properties['content_type'] == 'text':
            retrieval_str += (
                f"Source Document: {item.properties['source_document']}, \n"
                f"Page Number: {item.properties['page_number']}\n"
            )
            if item.properties['part_number'] is not None:
                retrieval_str += f"Part Number: {item.properties['part_number']}\n"
            retrieval_str += f"Text: {item.properties['text']}\n"
        elif item.properties['content_type'] == 'image':
            retrieval_str += (
                f"Source Document: {item.properties['source_document']}, \n"
                f"Page Number: {item.properties['page_number']}\n")
            retrieval_str += f"Description: {item.properties['description']}\n"
        elif item.properties['content_type'] == 'table':
            retrieval_str += (
                f"Source Document: {item.properties['source_document']}, "
                f"Page Number: {item.properties['page_number']}\n"
            )
            retrieval_str += f"Description: {item.properties['description']}\n"
            retrieval_str += f"Table in HTML: \n{item.properties['table_html']}\n"

        retrieval_str += "---\n"
    return retrieval_str, response_object_json_ls


def parse_search_response_objects(search_response_objects):
    # parse text and tables into strings, and for images get base64 to be attached to llm in prompt
    response_object_json_ls = []
    retrieval_str = ""
    for i, item in enumerate(search_response_objects, 1):
        object_properties_dict = copy.deepcopy(item.properties)
        object_properties_dict.update({
            "score": item.metadata.score, "explain_score": item.metadata.explain_score
        })
        response_object_json_ls.append(object_properties_dict)
        source_file_name = os.path.basename(os.path.normpath(item.properties['source_document']))
        # retrieval string text that can be passed directly to llm
        retrieval_str += (
            f"Retrieval Item number {i}, Type: {item.properties['content_type']}, "
            f"File category: {item.properties['file_category']}\n"
        )
        if item.properties['content_type'] == 'text':
            pg_num = item.properties['page_number'] if item.properties['page_number'] is not None \
                else item.properties['page_chunk']
            retrieval_str += (
                f"Source Document: {source_file_name}, Page Number: {pg_num}\n"
            )
            if item.properties['part_number'] is not None:
                retrieval_str += f"Part Number: {item.properties['part_number']}\n"
            retrieval_str += f"Text: {item.properties['text']}\n"
        elif item.properties['content_type'] == 'image':
            retrieval_str += (
                f"Source Document: {source_file_name}, Page Number: {item.properties['page_number']}\n")
            retrieval_str += f"Description: {item.properties['description']}\n"
        elif item.properties['content_type'] == 'table':
            retrieval_str += (
                f"Source Document: {source_file_name}, Page Number: {item.properties['page_number']}\n"
            )
            retrieval_str += f"Description: {item.properties['description']}\n"
            # check number of tokens to avoid over large tables
            num_tokens = num_tokens_from_string(item.properties['table_html'], "o200k_base")
            if num_tokens < 10000:
                retrieval_str += f"Table in HTML: \n{item.properties['table_html']}\n"

        retrieval_str += "---\n"
    return retrieval_str, response_object_json_ls


def parse_search_response_objects_str_img_only(search_response_objects):
    # parse text and tables into strings, and for images get base64 to be attached to llm in prompt
    response_object_json_ls = []
    retrieval_str = ""
    image_base64_ls = []
    for item in search_response_objects:
        retrieval_str += f"Type: {item.properties['content_type']}\n"
        if item.properties['content_type'] == 'text':
            retrieval_str += f"Source Document: {item.properties['source_document']}, Page Number: {item.properties['page_number']}\n"
            if item.properties['part_number'] is not None:
                retrieval_str += f"Part Number: {item.properties['part_number']}"
            retrieval_str += f"Text: {item.properties['text']}\n"
        elif item.properties['content_type'] == 'image':
            retrieval_str += f"Source Document: {item.properties['source_document']}, Page Number: {item.properties['page_number']}\n"
            retrieval_str += f"Description: {item.properties['description']}\n"
            if item.properties['base64_encoding'] is not None and item.properties['mime_type'] is not None:
                image_base64_ls.append({
                    "base64_encoding": item.properties['base64_encoding'],
                    "mime_type": item.properties['mime_type']
                })
        elif item.properties['content_type'] == 'table':
            retrieval_str += f"Source Document: {item.properties['source_document']}, Page Number: {item.properties['page_number']}\n"
            retrieval_str += f"Description: {item.properties['description']}\n"
            retrieval_str += f"Table in HTML: \n{item.properties['table_html']}\n"
            if item.properties['base64_encoding'] is not None and item.properties['mime_type'] is not None:
                image_base64_ls.append({
                    "base64_encoding": item.properties['base64_encoding'],
                    "mime_type": item.properties['mime_type']
                })
        retrieval_str += f"Score to query: {item.metadata.score:.3f}\n"
        retrieval_str += f"Explanation for the score: {item.metadata.explain_score}\n"
        retrieval_str += "---\n"
    return retrieval_str, image_base64_ls


def rag_search_and_retrieval(company_name, query, file_category=None, alpha=0.75, limit=20):
    company_name_in_vector_db = change_company_name_gsc_to_vectordb(company_name=company_name) #company_name.replace(" ", "_")
    client_weaviate = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    multi_collection_companies = client_weaviate.collections.get("Companies")
    company_tenant = multi_collection_companies.with_tenant(company_name_in_vector_db)

    search_objects = search_multimodal(
        company_tenant=company_tenant, query=query, file_category=file_category, alpha=alpha, limit=limit
    )
    retrieval_str = parse_search_response_objects(search_response_objects=search_objects)
    client_weaviate.close()
    print(f"retrieval_str: {retrieval_str}")
    return retrieval_str #, response_object_json_ls


def gemini_web_search(search_input: str):
    try:
        google_search_tool = types.Tool(
            google_search = types.GoogleSearch()
        )
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=search_input,
            config=types.GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"]
            )
        )

        # Extract the main response text
        response_text = response.text if hasattr(response, 'text') else "No response text found."
        
        # Build citation string from grounding metadata
        # Format the final output
        result = f"Web Search Result:\n{response_text}\n\n"
            
        return result
        
    except Exception as e:
        print(f"Error performing web search: {str(e)}")
        return "Error performing web search. Please try a different search query."


def get_company_financials_from_yfinance_api(company_symbol):
    # SSL SOL.JO SAO.F SAO.BE SAO.MU SAO.DU , 4321.SR , TLW.L TLW.XC TUWLF TQW.SG TQW.F TUWOY
    tk = yf.Ticker(company_symbol)
    income_data = tk.income_stmt.T
    bal_data = tk.balance_sheet.T
    cash_data = tk.cash_flow.T
    # add currency
    income_data["Currency"] = tk.info["currency"]
    bal_data["Currency"] = tk.info["currency"]
    cash_data["Currency"] = tk.info["currency"]
    return income_data, bal_data, cash_data


def get_company_financials_from_store_yf(company_name):
    # load files from gcs blob
    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

    blob_inc = bucket.blob(f'stored_api_data/{company_name}/income_data.csv')
    blob_bal = bucket.blob(f'stored_api_data/{company_name}/balance_data.csv')
    blob_cash = bucket.blob(f'stored_api_data/{company_name}/cash_data.csv')

    contents_inc = blob_inc.download_as_bytes()
    contents_bal = blob_bal.download_as_bytes()
    contents_cash = blob_cash.download_as_bytes()

    income_data = pd.read_csv(io.StringIO(contents_inc.decode("utf-8")), index_col=0)
    bal_data = pd.read_csv(io.StringIO(contents_bal.decode("utf-8")), index_col=0)
    cash_data = pd.read_csv(io.StringIO(contents_cash.decode("utf-8")), index_col=0)

    storage_client.close()
    return income_data, bal_data, cash_data


def aggregate_company_financials_yf_to_df(company_name, from_api=False):
    income_df, balance_df, cash_df = get_company_financials_from_yfinance_api(company_symbol="SSL") if from_api \
        else get_company_financials_from_store_yf(company_name=company_name)

    # make index into date column fiscalDateEnding
    income_df = income_df.reset_index(names="Fiscal Date Ending")
    balance_df = balance_df.reset_index(names="Fiscal Date Ending")
    cash_df = cash_df.reset_index(names="Fiscal Date Ending")

    suffixes = ('_income', '_balance')
    metric_df = income_df.merge(balance_df, left_on='Fiscal Date Ending', right_on='Fiscal Date Ending', suffixes=suffixes)
    repeat_cols = [col_name for col_name in metric_df.columns if suffixes[0] in col_name or suffixes[1] in col_name]
    original_cols = list(set([col.split('_')[0] for col in repeat_cols]))

    def map_cols(row, source_cols_left, source_cols_right):
        if row[source_cols_left] == row[source_cols_right]:
            val = row[source_cols_left]
        elif row[source_cols_left] == 'None' and row[source_cols_right] != 'None':
            val = row[source_cols_right]
        elif row[source_cols_right] == 'None' and row[source_cols_left] != 'None':
            val = row[source_cols_left]
        else:
            val = row[source_cols_right]
            # val = row[source_cols[0]]
        return val

    for col in original_cols:
        # metric_df[f'{colname}_income'] == metric_df[f'{colname}_balance']
        metric_df[col] = metric_df.apply(
            map_cols, args=(f'{col}{suffixes[0]}', f'{col}{suffixes[1]}'), axis=1
        )

    # drop repeated columns
    metric_df.drop(columns=repeat_cols, inplace=True)
    # merge cash data
    suffixes = ('_metric', '_cash')
    metric_df = metric_df.merge(cash_df, left_on='Fiscal Date Ending', right_on='Fiscal Date Ending', suffixes=suffixes)
    repeat_cols = [col_name for col_name in metric_df.columns if suffixes[0] in col_name or suffixes[1] in col_name]
    original_cols = list(set([col.split('_')[0] for col in repeat_cols]))

    for col in original_cols:
        # metric_df[f'{colname}_income'] == metric_df[f'{colname}_balance']
        metric_df[col] = metric_df.apply(
            map_cols, args=(f'{col}{suffixes[0]}', f'{col}{suffixes[1]}'), axis=1
        )

    # drop repeated columns
    metric_df.drop(columns=repeat_cols, inplace=True)
    return metric_df


def get_financial_metrics_yf_api_for_table(company_name, from_api=False):
    # get financial metrics from Alpha Vantage API
    csv_str = ""
    if company_name == "Sasol Limited":
        csv_alp_str = get_financial_metrics_api_for_table(company_name=company_name, from_api=from_api)
        csv_str += f"Financial metrics from Alpha Vantage API: \n{csv_alp_str} \n\n"
    # get financial metrics from yfinance API
    df = aggregate_company_financials_yf_to_df(company_name=company_name, from_api=from_api)
    df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
    df['Debt-To-EBITDA'] = df['Total Debt'] / df['EBITDA']
    df['Net Debt-To-EBITDA'] = df['Net Debt'] / df['EBITDA']
    df['Interest Coverage Ratio'] = df['EBIT'] / df['Interest Expense']
    df["EBITDA Margin"] = (df['EBITDA'] / df['Total Revenue']) * 100
    df["Net Income Margin"] = (df["Net Income"] / df['Total Revenue']) * 100
    # select only metrics needed by table
    metrics_sel = (
        "Fiscal Date Ending", "Total Revenue", "EBITDA", "EBIT", "Net Income", "Cash And Cash Equivalents",
        "Current Assets", "Current Liabilities", "Total Debt", "Interest Expense",
        "Current Ratio", "Net Debt", "Debt-To-EBITDA", "Net Debt-To-EBITDA", "Interest Coverage Ratio",
        "EBITDA Margin", "Net Income Margin"
    )
    df_select = df.loc[:, metrics_sel]
    metric_df_csv_str = df_select.to_csv(index=False)
    # combine csv strings
    csv_str += f"Financial metrics from Yahoo Finance API: \n{metric_df_csv_str}"
    return csv_str


def get_financial_metrics_yf_api(company_name, from_api=False):
    csv_str = ""
    if company_name == "Sasol Limited":
        # get financial metrics from Alpha Vantage API
        csv_alp_str = get_financial_metrics_api_for_chatbot(company_name=company_name, from_api=from_api)
        csv_str += f"Financial metrics from Alpha Vantage API: \n{csv_alp_str} \n\n"

    # get financial metrics from yfinance API
    df = aggregate_company_financials_yf_to_df(company_name=company_name, from_api=from_api)
    df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
    df['Debt-To-EBITDA'] = df['Total Debt'] / df['EBITDA']
    df['Net Debt-To-EBITDA'] = df['Net Debt'] / df['EBITDA']
    df['Interest Coverage Ratio'] = df['EBIT'] / df['Interest Expense']
    df["EBITDA Margin"] = (df['EBITDA'] / df['Total Revenue']) * 100
    df["Net Income Margin"] = (df["Net Income"] / df['Total Revenue']) * 100
    metric_df_csv_str = df.to_csv(index=False)
    # combine csv strings
    csv_str += f"Financial metrics from Yahoo Finance API: \n{metric_df_csv_str}"
    return csv_str


def get_company_financials_from_alpha_vantage_api(company_symbol):
    # call for AlphaVantage API to get the table figures
    # get financial metrics from Alpha Vantage API
    income_data = requests.get(
        (f"https://www.alphavantage.co/query?function=INCOME_STATEMENT"
         f"&symbol={company_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}")
    ).json()
    bal_data = requests.get(
        (f"https://www.alphavantage.co/query?function=BALANCE_SHEET"
         f"&symbol={company_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}")
    ).json()
    cash_data = requests.get(
        (f"https://www.alphavantage.co/query?function=CASH_FLOW"
         f"&symbol={company_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}")
    ).json()
    return income_data, bal_data, cash_data


def get_company_financials_from_store(company_name):
    # load files from gcs blob
    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

    blob_inc = bucket.blob(f'stored_api_data/{company_name}/income_data.json')
    blob_bal = bucket.blob(f'stored_api_data/{company_name}/balance_data.json')
    blob_cash = bucket.blob(f'stored_api_data/{company_name}/cash_data.json')

    contents_inc = blob_inc.download_as_bytes()
    contents_bal = blob_bal.download_as_bytes()
    contents_cash = blob_cash.download_as_bytes()

    income_data = json.loads(contents_inc.decode("utf-8"))
    bal_data = json.loads(contents_bal.decode("utf-8"))
    cash_data = json.loads(contents_cash.decode("utf-8"))

    storage_client.close()
    return income_data, bal_data, cash_data


def aggregate_company_financials_to_df(company_name, from_api=False):
    income_data, bal_data, cash_data = get_company_financials_from_alpha_vantage_api(company_symbol="SSL") if from_api \
        else get_company_financials_from_store(company_name=company_name)

    income_df = pd.DataFrame(data={
        key: list(value) for key, value in zip(
            income_data['quarterlyReports'][0].keys(),
            zip(*[dict.values() for dict in income_data['quarterlyReports'][:20]])
        )
    })
    balance_df = pd.DataFrame(data={
        key: list(value) for key, value in zip(
            bal_data['quarterlyReports'][0].keys(),
            zip(*[dict.values() for dict in bal_data['quarterlyReports'][:20]])
        )
    })
    cash_df = pd.DataFrame(data={
        key: list(value) for key, value in zip(
            cash_data['quarterlyReports'][0].keys(),
            zip(*[dict.values() for dict in cash_data['quarterlyReports'][:20]])
        )
    })

    suffixes = ('_income', '_balance')
    metric_df = income_df.merge(balance_df, left_on='fiscalDateEnding', right_on='fiscalDateEnding', suffixes=suffixes)
    repeat_colnames = [col_name for col_name in metric_df.columns if suffixes[0] in col_name or suffixes[1] in col_name]
    original_colnames = list(set([colname.split('_')[0] for colname in repeat_colnames]))

    def map_cols(row, source_colname_left, source_colname_right):
        if row[source_colname_left] == row[source_colname_right]:
            val = row[source_colname_left]
        elif row[source_colname_left] == 'None' and row[source_colname_right] != 'None':
            val = row[source_colname_right]
        elif row[source_colname_right] == 'None' and row[source_colname_left] != 'None':
            val = row[source_colname_left]
        else:
            val = row[source_colname_right]
            # val = row[source_colnames[0]]
        return val

    for colname in original_colnames:
        # metric_df[f'{colname}_income'] == metric_df[f'{colname}_balance']
        metric_df[colname] = metric_df.apply(
            map_cols, args=(f'{colname}{suffixes[0]}', f'{colname}{suffixes[1]}'), axis=1
        )

    # drop repeated columns
    metric_df.drop(columns=repeat_colnames, inplace=True)
    # merge cash data
    suffixes = ('_metric', '_cash')
    metric_df = metric_df.merge(cash_df, left_on='fiscalDateEnding', right_on='fiscalDateEnding', suffixes=suffixes)
    repeat_colnames = [col_name for col_name in metric_df.columns if suffixes[0] in col_name or suffixes[1] in col_name]
    original_colnames = list(set([colname.split('_')[0] for colname in repeat_colnames]))

    for colname in original_colnames:
        # metric_df[f'{colname}_income'] == metric_df[f'{colname}_balance']
        metric_df[colname] = metric_df.apply(
            map_cols, args=(f'{colname}{suffixes[0]}', f'{colname}{suffixes[1]}'), axis=1
        )

    # drop repeated columns
    metric_df.drop(columns=repeat_colnames, inplace=True)
    return metric_df


def get_financial_metrics_api_for_table(company_name, from_api=False):
    df = aggregate_company_financials_to_df(company_name=company_name, from_api=from_api)

    numeric_col_names = df.columns.drop('fiscalDateEnding')
    df[numeric_col_names] = df[numeric_col_names].apply(pd.to_numeric, errors='coerce')

    df['Current Ratio'] = df['totalCurrentAssets'] / df['totalCurrentLiabilities']
    df['Net Debt'] = df['shortLongTermDebtTotal'] - df['cashAndCashEquivalentsAtCarryingValue']
    df['Debt-to-ebitda'] = df['shortLongTermDebtTotal'] / df['ebitda']
    df['Net Debt-to-ebitda'] = df['Net Debt'] / df['ebitda']
    df['Interest Coverage Ratio'] = df['ebit'] / df['interestExpense']
    df["EBITDA Margin"] = (df['ebitda'] / df['totalRevenue']) * 100
    df["Net Income Margin"] = (df["netIncome"] / df['totalRevenue']) * 100

    df = df.rename(
        columns={
            "fiscalDateEnding": "Fiscal Date Ending", "totalRevenue": "Total Revenue", "netIncome": "Net Income",
            "shortLongTermDebtTotal": "Gross Debt", "cashAndCashEquivalentsAtCarryingValue": "Cash And Equivalents",
            "totalCurrentAssets": "Total Current Assets", "totalCurrentLiabilities": "Total Current Liabilities",
            "interestExpense": "Interest Expense",
        }
    )
    # dates_select = [latest_fiscal_date, previous_year_fiscal_date, previous_half_year_fiscal_date]
    metrics_select = (
        "Fiscal Date Ending", "Total Revenue", "ebitda", "ebit", "Net Income", "Cash And Equivalents",
        "Total Current Assets", "Total Current Liabilities", "Gross Debt", "Interest Expense",
        "Current Ratio", "Net Debt", "Debt-to-ebitda", "Net Debt-to-ebitda", "Interest Coverage Ratio",
        "EBITDA Margin", "Net Income Margin"
    )
    df_select = df.loc[:, metrics_select]
    metric_df_csv_str = df_select.to_csv(index=False)
    return metric_df_csv_str


def get_financial_metrics_api(
        company_name, latest_fiscal_date, previous_year_fiscal_date, previous_half_year_fiscal_date, from_api=False
):
    df = aggregate_company_financials_to_df(company_name=company_name, from_api=from_api)

    numeric_col_names = df.columns.drop('fiscalDateEnding')
    df[numeric_col_names] = df[numeric_col_names].apply(pd.to_numeric, errors='coerce')

    df['Current Ratio'] = df['totalCurrentAssets'] / df['totalCurrentLiabilities']
    df['Net Debt'] = df['shortLongTermDebtTotal'] - df['cashAndCashEquivalentsAtCarryingValue']
    df['Debt-to-ebitda'] = df['shortLongTermDebtTotal'] / df['ebitda']
    df['Net Debt-to-ebitda'] = df['Net Debt'] / df['ebitda']
    df['Interest Coverage Ratio'] = df['ebit'] / df['interestExpense']
    df["EBITDA Margin"] = (df['ebitda'] / df['totalRevenue']) * 100
    df["Net Income Margin"] = (df["netIncome"] / df['totalRevenue']) * 100

    df = df.rename(
        columns={
            "fiscalDateEnding": "Fiscal Date Ending", "totalRevenue": "Total Revenue", "netIncome": "Net Income",
            "shortLongTermDebtTotal": "Gross Debt", "cashAndCashEquivalentsAtCarryingValue": "Cash And Equivalents",
            "totalCurrentAssets": "Total Current Assets", "totalCurrentLiabilities": "Total Current Liabilities",
            "interestExpense": "Interest Expense",
        }
    )
    dates_select = [latest_fiscal_date, previous_year_fiscal_date, previous_half_year_fiscal_date]
    metrics_select = (
        "Fiscal Date Ending", "Total Revenue", "ebitda", "ebit", "Net Income", "Cash And Equivalents",
        "Total Current Assets", "Total Current Liabilities", "Gross Debt", "Interest Expense",
        "Current Ratio", "Net Debt", "Debt-to-ebitda", "Net Debt-to-ebitda", "Interest Coverage Ratio",
        "EBITDA Margin", "Net Income Margin"
    )
    df_select = df.loc[df["Fiscal Date Ending"].isin(dates_select), metrics_select]
    metric_df_csv_str = df_select.to_csv(index=False)
    return metric_df_csv_str


def get_financial_metrics_api_for_chatbot(company_name, from_api=False):
    df = aggregate_company_financials_to_df(company_name=company_name, from_api=from_api)
    # turn to numerics for calculations
    numeric_col_names = df.columns.drop('fiscalDateEnding')
    df[numeric_col_names] = df[numeric_col_names].apply(pd.to_numeric, errors='coerce')
    # add new columns for common metrics
    df['currentRatio'] = df['totalCurrentAssets'] / df['totalCurrentLiabilities']
    df['netDebt'] = df['shortLongTermDebtTotal'] - df['cashAndCashEquivalentsAtCarryingValue']
    df['debtToEbitdaRatio'] = df['shortLongTermDebtTotal'] / df['ebitda']
    df['netDebtToEbitdaRatio'] = df['netDebt'] / df['ebitda']
    df['interestCoverageRatio'] = df['ebit'] / df['interestExpense']
    df["ebitdaMargin"] = (df['ebitda'] / df['totalRevenue']) * 100
    df["netIncomeMargin"] = (df["netIncome"] / df['totalRevenue']) * 100
    # rename columns for more explicit names
    df = df.rename(
        columns={
            "fiscalDateEnding": "Fiscal Date Ending", "totalRevenue": "Total Revenue", "netIncome": "Net Income",
            "shortLongTermDebtTotal": "Gross Debt", "cashAndCashEquivalentsAtCarryingValue": "Cash And Equivalents",
            "totalCurrentAssets": "Total Current Assets", "totalCurrentLiabilities": "Total Current Liabilities",
            "interestExpense": "Interest Expense",
        }
    )
    # convert csv to string
    metric_df_csv_str = df.to_csv(index=False)
    return metric_df_csv_str


def get_company_bonds_from_cbonds_api(company_name):
    # cbonds_sasol_data = resp_data.json()
    with requests.Session() as sess:
        c = sess.get(
            f"https://ws.cbonds.info/services/json?login={os.getenv('CBONDS_LOGIN_EMAIL')}&password={os.getenv('CBONDS_PASSWORD')}&lang=eng")
        json_data_emits = {
            "auth": {"login": f"{os.getenv('CBONDS_LOGIN_EMAIL')}", "password": f"{os.getenv('CBONDS_PASSWORD')}"},
            "filters": [{"field": "full_name_eng", "operator": "eq", "value": f"{company_name}"}],
            "quantity": {"limit": 10, "offset": 0}, "sorting": [{"field": "", "order": "asc"}],
            "fields": [
                {"field": "id"}, {"field": "bik"}, {"field": "branch_id"},
                {"field": "branch_name_eng"}, {"field": "branch_name_ita"},
                {"field": "branch_name_pol"}, {"field": "branch_name_rus"},
                {"field": "code_of_emitent"}, {"field": "country_id"},
                {"field": "country_name_eng"}, {"field": "country_name_ita"},
                {"field": "country_name_pol"}, {"field": "country_name_rus"},
                {"field": "country_region_id"}, {"field": "country_subregion_id"},
                {"field": "email"}, {"field": "emitents_id_absorption"},
                {"field": "emitent_address_eng"}, {"field": "emitent_address_ita"},
                {"field": "emitent_address_pol"}, {"field": "emitent_address_rus"},
                {"field": "emitent_inn"}, {"field": "emitent_inn_ua"},
                {"field": "emitent_kpp"}, {"field": "emitent_lei"}, {"field": "emitent_spv"},
                {"field": "fax"}, {"field": "full_inn"}, {"field": "full_name_eng"},
                {"field": "full_name_ita"}, {"field": "full_name_pol"},
                {"field": "full_name_rus"}, {"field": "legal_address_eng"},
                {"field": "legal_address_ita"}, {"field": "legal_address_pol"},
                {"field": "legal_address_rus"}, {"field": "loroaccount"},
                {"field": "name_eng"}, {"field": "name_ita"}, {"field": "name_pol"},
                {"field": "name_rus"}, {"field": "ogrn"}, {"field": "ogrn_date"},
                {"field": "ogrn_first_date"}, {"field": "okato"}, {"field": "okpo"},
                {"field": "okved"}, {"field": "phone"}, {"field": "profile_eng"},
                {"field": "profile_ita"}, {"field": "profile_pol"}, {"field": "profile_rus"},
                {"field": "regsn"}, {"field": "reg_form_id"}, {"field": "reg_form_name_eng"},
                {"field": "reg_form_name_ita"}, {"field": "reg_form_name_pol"},
                {"field": "reg_form_name_rus"}, {"field": "significant_emitents_subr_1"},
                {"field": "site_eng"}, {"field": "site_ita"}, {"field": "site_pol"},
                {"field": "site_rus"}, {"field": "swift"}, {"field": "type_id"},
                {"field": "type_name_eng"}, {"field": "type_name_ita"},
                {"field": "type_name_pol"}, {"field": "type_name_rus"},
                {"field": "updating_date"}, {"field": "registration_country_id"},
                {"field": "registration_country_rus"}, {"field": "registration_country_eng"},
                {"field": "registration_country_pol"}, {"field": "registration_country_ita"},
                {"field": "icb"}, {"field": "icb_name_rus"}, {"field": "icb_name_eng"},
                {"field": "id_group"}, {"field": "group_name_rus"}, {"field": "sic"},
                {"field": "sic_name_eng"}, {"field": "nace"}, {"field": "nace_name_eng"},
                {"field": "is_emitent"}, {"field": "bad_emitent"}, {"field": "show_ru"},
                {"field": "cik"}, {"field": "is_international_emitent"}, {"field": "naics"},
                {"field": "naics_name_eng"}, {"field": "is_country"},
                {"field": "emitent_statuses_id"}, {"field": "has_unsettled_default"},
                {"field": "emitent_categories_id"}
            ]
        }
        emits_data = sess.get(
            url="https://ws.cbonds.info/services/json/get_emitents/?lang=eng", json=json_data_emits
        ).json()
        company_emits = [itm for itm in emits_data['items'] if itm['full_name_eng'] == company_name]
        # todo: make sure length of company emits is 1, otherwise there's company name ambiguity
        # if len(company_emitent) == 1:
        company_emits_id = company_emits[0]['id']
        json_data_emissions = {
            "auth": {"login": f"{os.getenv('CBONDS_LOGIN_EMAIL')}", "password": f"{os.getenv('CBONDS_PASSWORD')}"},
            "filters": [{"field": "emitent_id", "operator": "eq", "value": f"{company_emits_id}"}],
            "quantity": {"limit": 10, "offset": 0}, "sorting": [{"field": "", "order": "asc"}],
            "fields": [
                {"field": "id"}, {"field": "formal_emitent_id"}, {"field": "bbgid"},
                {"field": "bbgid_ticker"}, {"field": "cfi_code"}, {"field": "cfi_code_144a"},
                {"field": "convertable"}, {"field": "coupon_type_id"},
                {"field": "updating_date"}, {"field": "cupon_period"},
                {"field": "currency_id"}, {"field": "currency_name"}, {"field": "cusip_144a"},
                {"field": "cusip_regs"}, {"field": "date_of_end_placing"},
                {"field": "date_of_start_circulation"}, {"field": "date_of_start_placing"},
                {"field": "document_eng"}, {"field": "document_rus"},
                {"field": "document_pol"}, {"field": "document_ita"},
                {"field": "early_redemption_date"}, {"field": "emission_cupon_basis_id"},
                {"field": "emission_cupon_basis_title"}, {"field": "emission_emitent_id"},
                {"field": "emitent_id"}, {"field": "emitent_branch_id"},
                {"field": "emitent_branch_name_eng"}, {"field": "emitent_branch_name_rus"},
                {"field": "floating_rate"}, {"field": "guarantor_id"},
                {"field": "integral_multiple"}, {"field": "isin_code"},
                {"field": "isin_code_144a"}, {"field": "isin_code_3"}, {"field": "kind_id"},
                {"field": "margin"}, {"field": "maturity_date"}, {"field": "micex_shortname"},
                {"field": "nominal_price"}, {"field": "initial_nominal_price"},
                {"field": "number_of_emission"}, {"field": "number_of_emission_eng"},
                {"field": "outstanding_nominal_price"}, {"field": "reference_rate_id"},
                {"field": "reference_rate_name_eng"}, {"field": "reference_rate_name_rus"},
                {"field": "registration_date"}, {"field": "settlement_date"},
                {"field": "state_reg_number"}, {"field": "status_id"},
                {"field": "subkind_id"}, {"field": "subordinated_debt"}, {"field": "vid_id"},
                {"field": "cupon_eng"}, {"field": "cupon_rus"}, {"field": "kind_name_eng"},
                {"field": "kind_name_rus"}, {"field": "kind_name_pol"},
                {"field": "kind_name_ita"}, {"field": "emitent_country"},
                {"field": "emitent_country_region_id"},
                {"field": "emitent_country_subregion_id"}, {"field": "emitent_name_eng"},
                {"field": "emitent_name_rus"}, {"field": "offert_date"},
                {"field": "offert_date_put"}, {"field": "offert_date_call"},
                {"field": "update_time"}, {"field": "mortgage_bonds"},
                {"field": "restructing"}, {"field": "restructing_date"},
                {"field": "perpetual"}, {"field": "remaining_outstand_amount"},
                {"field": "secured_debt"}, {"field": "eurobonds_nominal"},
                {"field": "structured_note"}, {"field": "emitent_inn"},
                {"field": "formal_emitent_name_rus"}, {"field": "emitent_type_name_rus"},
                {"field": "emitent_type"}, {"field": "announced_volume_new"},
                {"field": "placed_volume_new"}, {"field": "outstanding_volume"},
                {"field": "indexation_eng"}, {"field": "indexation_rus"},
                {"field": "order_book"}, {"field": "number_of_trades_on_issue_date"},
                {"field": "emitent_type_name_eng"}, {"field": "emitent_country_name_eng"},
                {"field": "emitent_country_name_rus"}
            ]
        }
        emissions_data = sess.get(
            url="https://ws.cbonds.info/services/json/get_emissions/?lang=eng", json=json_data_emissions
        ).json()

    sess.close()
    return emissions_data


def get_company_bonds_from_store(company_name):
    # load files from gcs blob
    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))
    blob = bucket.blob(f'stored_api_data/{company_name}/bond_data.json')
    contents = blob.download_as_bytes()
    bond_data = json.loads(contents.decode("utf-8"))
    storage_client.close()
    return bond_data


def aggregate_company_bonds_to_df(company_name, from_api=False):
    emissions_data = get_company_bonds_from_cbonds_api(company_name=company_name) if from_api \
        else get_company_bonds_from_store(company_name=company_name)
    emissions_df = pd.DataFrame(emissions_data['items'])
    return emissions_df


def get_bond_data_api(company_name, date_today, from_api=False):
    df = aggregate_company_bonds_to_df(company_name=company_name, from_api=from_api)
    df = df.sort_values(by='maturity_date', ascending=False)
    df = df.rename(columns={
        "document_eng": "bond_name",
        "cupon_eng": "coupon_rate",
        "kind_name_eng": "bond_kind_name",
        "remaining_outstand_amount": "amount"
    })
    bond_columns = (
        "id", "currency_name", "bond_name", "maturity_date", "settlement_date", "coupon_rate", "bond_kind_name",
        "offert_date", "amount"
    )
    bond_df = df.loc[df['maturity_date'] > date_today, bond_columns] #.to_csv(index=False)
    bond_df_csv_str = bond_df.to_csv(index=False)
    return bond_df_csv_str


def get_bond_data_api_for_chatbot(company_name, from_api=False):
    df = aggregate_company_bonds_to_df(company_name=company_name, from_api=from_api)
    # bond_columns = (
    #     "id", "currency_name", "document_eng", "maturity_date", "settlement_date", "cupon_eng", "kind_name_eng",
    #     "offert_date", "remaining_outstand_amount"
    # )
    df = df.sort_values(by='maturity_date', ascending=False)
    df = df.rename(columns={
        "document_eng": "bond_name",
        "cupon_eng": "coupon_rate",
        "kind_name_eng": "bond_kind_name",
        "remaining_outstand_amount": "amount"
    })
    bond_df_csv_str = df.to_csv(index=False)
    return bond_df_csv_str


def process_tool_call(tool_name, tool_input):
    if tool_name == "rag_search_and_retrieval":
        return rag_search_and_retrieval(**tool_input)
    elif tool_name == "gemini_web_search":
        return gemini_web_search(**tool_input)
    elif tool_name == "get_financial_metrics_yf_api_for_table":
        return get_financial_metrics_yf_api_for_table(**tool_input)
    elif tool_name == "get_financial_metrics_yf_api":
        return get_financial_metrics_yf_api(**tool_input)
    elif tool_name == "get_financial_metrics_api":
        return get_financial_metrics_api(**tool_input)
    elif tool_name == "get_financial_metrics_api_for_chatbot":
        return get_financial_metrics_api_for_chatbot(**tool_input)
    elif tool_name == "get_bond_data_api":
        return get_bond_data_api(**tool_input)
    elif tool_name == "get_bond_data_api_for_chatbot":
        return get_bond_data_api_for_chatbot(**tool_input)

def gemini_pro_loop_with_tools(
        model_name, input_messages, tools, file_category=None, company_name=None
):
    continue_function_call = True
    step_number = 0
    step_responses = []
    function_declarations = []
    for tool in tools:
        function_declarations.append(tool)

    # Create Tool list for Gemini (only one Tool with all function declarations)
    gemini_tools = [types.Tool(function_declarations=function_declarations)] if function_declarations else []

    while continue_function_call:
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in input_messages:
            if msg.get('type') == 'function_call':
                gemini_messages.append(types.Content(
                    role='model',
                    parts=[types.Part(function_call=types.FunctionCall(
                        name=msg['name'],
                        args=json.loads(msg['arguments'])
                    ))]
                ))
            elif msg.get('type') == 'function_call_output':
                gemini_messages.append(types.Content(
                    role='user',
                    parts=[types.Part.from_function_response(
                        name=msg.get('name', ''),  # Use empty string as fallback if name not provided
                        response={"result": msg['output']}
                    )]
                ))
            else:
                role = 'user' if msg['role'] in ['user', 'system'] else 'model'
                content = str(msg['content']) if isinstance(msg['content'], (list, dict)) else msg['content']
                gemini_messages.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)]
                ))

        # Configure tool usage
        tool_config_mode = 'ANY' if step_number < len(tools) else 'AUTO'
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=tool_config_mode)
        )
        config = types.GenerateContentConfig(
            tools=gemini_tools,
            tool_config=tool_config
        )

        print("----------gemini_messages", gemini_messages)
        response = client_gemini.models.generate_content(
            model=model_name,
            contents=gemini_messages,
            config=config
        )
        step_number += 1

        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            print(f"Function to call: {function_call}")
            func_step_resp_ls = []
        
            tool_name = function_call.name
            tool_args = function_call.args
            if tool_name == "rag_search_and_retrieval":
                if company_name is not None:
                    tool_args.update({"company_name": company_name})
                if file_category is not None:
                    tool_args.update({"file_category": file_category})
            elif (tool_name == "get_financial_metrics_yf_api_for_table" or
                    tool_name == "get_financial_metrics_yf_api" or
                    tool_name == "get_financial_metrics_api" or
                    tool_name == "get_financial_metrics_api_for_chatbot"):
                if company_name is not None:
                    tool_args.update({"company_name": company_name})
                tool_args.update({"from_api": False})
            elif tool_name == "get_bond_data_api" or tool_name == "get_bond_data_api_for_chatbot":
                if company_name is not None:
                    tool_args.update({"company_name": company_name})
                tool_args.update({"from_api": False})

            function_result = process_tool_call(tool_name=tool_name, tool_input=tool_args)
            
            print("Function calling result", function_result)
            input_messages.append({
                "type": "function_call",
                "name": tool_name,
                "arguments": json.dumps(tool_args)
            })
            input_messages.append({
                "type": "function_call_output",
                "output": str(function_result[0] if tool_name == "rag_search_and_retrieval" else function_result)
            })
            if tool_name == "rag_search_and_retrieval":
                input_file_data_ls = []
                for obj in function_result[1]:
                    if ((obj['content_type'] == "image" or obj['content_type'] == 'table')
                            and 'mime_type' in obj.keys() and 'base64_encoding' in obj.keys()):
                        if obj['mime_type'] is not None and obj['base64_encoding'] is not None:
                            input_file_data_ls.append({
                                "type": "input_file",
                                "filename": os.path.basename(os.path.normpath(obj['source_document'])),
                                "file_data": f"data:{obj['mime_type']};base64,{obj['base64_encoding']}",
                            })
                input_messages.append({"role": "user", "content": input_file_data_ls})
            # append intermediate func calling steps
            func_step_resp_ls.append({
                "step_number": str(step_number),
                "calling": str(f"{tool_name}(**{json.dumps(tool_args)})"),
                "output": str(function_result)
            })
            step_responses.extend(func_step_resp_ls)
        else:
            print("No function call found in the response.")
            print(response.text)
            response_str = response.text
            step_responses.append({
                "step_number": str(step_number),
                "calling": "final",
                "output": response_str
            })
            continue_function_call = False
            break

    return response_str, step_responses

class ReasoningToolSchema:
    def __init__(self):
        # Define tools according to google.generativeai FunctionDeclaration schema
        self.rag_retrieval_tool = types.FunctionDeclaration(
            name="rag_search_and_retrieval",
            description=(
                "Search and retrieves context text chunks, images and tables that are relevant to the query from "
                "the vector database. Returns a string that parses the chunks together, and a list of dictionary "
                "where each dictionary contains the context details."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": (
                            "The company name to be provided to search within the company's tenant of the "
                            "Vector Database multi-tenant collection."
                        )
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query to be sent to the Vector Database tenant associated to the company, "
                            "for knowledge and context retrieval."
                        )
                    },
                },
                "required": [
                    "company_name",
                    "query"
                ]
                # additionalProperties removed
            }
        )
        self.gemini_web_search_tool = types.FunctionDeclaration(
            name="gemini_web_search",
            description=(
                "Do web search to retrieve online information and context to complete tasks. "
                "Returns text response according to web search, and a list of citations for the website used for reference."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "search_input": {
                        "type": "string",
                        "description": (
                            "The search query where the web search tool will look up for results under this search query."
                        )
                    },
                },
                "required": [
                    "search_input",
                ]
                # additionalProperties removed
            }
        )
        self.get_financial_metrics_yf_api_for_table_tool = types.FunctionDeclaration(
            name="get_financial_metrics_yf_api_for_table",
            description=(
                "Extract a company's financial metric data for a given company name in a table format with "
                "the rows representing different fiscal date ending period, and the columns represents "
                "different financial metrics. This function returns text CSV string response with the financial metric "
                "columns given below extracted from API. The financial metrics available are in these columns "
                "with self-explanatory metric column names: \nFiscal Date Ending, Total Revenue, EBITDA, EBIT, "
                "Net Income, Cash And Cash Equivalents, Current Assets, Current Liabilities, Total Debt, "
                "Interest Expense, Current Ratio, Net Debt, Debt-To-EBITDA, Net Debt-to-EBITDA, "
                "Interest Coverage Ratio, EBITDA Margin, Net Income Margin"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": "The company name where financial data will be extracted for."
                    },
                },
                "required": [
                    "company_name",
                ]
                # additionalProperties removed
            }
        )
        self.get_financial_metrics_yf_api_tool = types.FunctionDeclaration(
            name="get_financial_metrics_yf_api",
            description=(
                "Extract a company's financial metric data for a given company name in a table format with "
                "the rows representing different fiscal date ending period, and the columns represents "
                "different financial metrics. This function returns text CSV string response with the financial metric "
                "columns given below extracted from API. The financial metrics available are in these columns "
                "with self-explanatory metric column names: \nFiscal Date Ending, Tax Effect Of Unusual Items, "
                # ... (rest of long description) ...
                "Net Income From Continuing Operations, Currency \n\n"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": (
                            "The company name where financial data will be extracted for."
                        )
                    },
                },
                "required": [
                    "company_name",
                ]
                # additionalProperties removed
            }
        )
        self.get_financial_metrics_api_tool = types.FunctionDeclaration(
            name="get_financial_metrics_api",
            description=(
                "Extract a company's financial metric data for a given company name and some given fiscal dates. "
                "Returns text CSV string response according to financial data extracted from API."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": (
                            "The company name where financial data will be extracted for."
                        )
                    },
                    "latest_fiscal_date": {
                        "type": "string",
                        "description": (
                            "The fiscal date ending associated with the latest half-year identified for this credit report."
                        )
                    },
                    "previous_year_fiscal_date": {
                        "type": "string",
                        "description": (
                            "The fiscal date ending associated with the same half-year one-year ago as compared to "
                            "the latest half-year identified for this credit report."
                        )
                    },
                    "previous_half_year_fiscal_date": {
                        "type": "string",
                        "description": (
                            "The fiscal date ending associated with the second-latest half-year right before "
                            "the latest half-year identified for this credit report."
                        )
                    },
                },
                "required": [
                    "company_name",
                    "latest_fiscal_date",
                    "previous_year_fiscal_date",
                    "previous_half_year_fiscal_date"
                ]
                # additionalProperties removed
            }
        )
        self.get_financial_metrics_api_for_chatbot_tool = types.FunctionDeclaration(
            name="get_financial_metrics_api_for_chatbot",
            description=(
                "Extract a company's financial metric data for a given company name in a table format with "
                "the rows representing different fiscal date ending period, and the columns represents "
                "different financial metrics. This function returns text CSV string response with the financial metric "
                "columns given below extracted from API. The financial metrics available are in these columns "
                "with self-explanatory metric column names: \nfiscalDateEnding, grossProfit, totalRevenue, "
                 # ... (rest of long description) ...
                "netDebtToEbitdaRatio, interestCoverageRatio, ebitdaMargin, netIncomeMargin \n\n"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": (
                            "The company name where financial data will be extracted for."
                        )
                    },
                },
                "required": [
                    "company_name",
                ]
                # additionalProperties removed
            }
        )
        self.get_bonds_api_tool = types.FunctionDeclaration(
            name="get_bond_data_api",
            description=(
                "Extract a company's bonds data for a given company name in a table format with the rows "
                "representing different bond instruments, and the columns represents different information "
                "of a particular bond. This function returns text CSV string response with the bond detail "
                "columns given below extracted from API. The bond information available are in these columns "
                "with self-explanatory metric column names: \nid, currency_name, bond_name, maturity_date, "
                "settlement_date, coupon_rate, bond_kind_name, offert_date, amount"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": (
                            "The company name where all bond data will be extracted for."
                        )
                    },
                    "date_today": {
                        "type": "string",
                        "description": (
                            "The date of today in anywhere-on-earth that is found out by doing web search. "
                            "The date MUST be in the format of YYYY-MM-DD, where YYYY is the four digits "
                            "representing year, MM is the two digits representing month, and DD is the two digits "
                            "representing day."
                        )
                    },
                },
                "required": [
                    "company_name",
                    "date_today",
                ]
                # additionalProperties removed
            }
        )
        self.get_bonds_api_for_chatbot_tool = types.FunctionDeclaration(
            name="get_bond_data_api_for_chatbot",
            description=(
                "Extract a company's bonds data for a given company name in a table format with the rows "
                "representing different bond instruments, and the columns represents different information "
                "of a particular bond. This function returns text CSV string response with the bond detail "
                "columns given below extracted from API. The bond information available are in these columns "
                "with self-explanatory metric column names: \nid, formal_emitent_id, bbgid, bbgid_ticker, cfi_code, "
                 # ... (rest of long description) ...
                "emitent_country_name_rus"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "enum": [
                            "Sasol Limited",
                            "Arabian Centres",
                            "Tullow Oil"
                        ],
                        "description": (
                            "The company name where all bond data will be extracted for."
                        )
                    },
                },
                "required": [
                    "company_name",
                ]
                # additionalProperties removed
            }
        )


def gcloud_connect_credentials():
    return service_account.Credentials.from_service_account_info(
        {
        "type": "service_account",
        "project_id": "credit-ai-project-456311",
        "private_key_id": "b6f4c7790995f0e2177389609ae15d7a760942e0",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCL7HhgcU2KJCUn\nBUbSBywRgknYf++iu3qmZCic1W4A3ERtESKdRFA/FWuFE4uccb4Fm0e5f/viuYx0\nds7Khqj1Nbbsa4ZWMlpKALxB3DVgzKJFbKkCIc7s61oZeLvopsRxB2xSdzgES/vT\nTx8J9D6LoYnBqlVjJBWSZDUbPKAmDA5C1/qamd8HnMMMYI8zJDLWCl5COebFGfZW\nnI/sQNLw+wLbl9dqcdGA4wFeWTaoaT8kIgkc7q2qpJoJVe4CRbwb3E7PAaGc6vx1\nfy4HBPHn4ZdRRSrlZ3Nu7TOvWuCDS4BQLVBwvTlT41xMHzR3YGboBhBW3zuqrQhS\ngq4phE+nAgMBAAECggEAA1AOUYo2yCYVUxhgfHYUm9C2QUMi80fCnmnePeHq5s5Q\nOGmgpJFXedOrmTNedme8lF/NK1EiYrqURRwC+iLroR9fPOp9L1E/d5ao3lsbHTcu\nQe9R2Qd5exZ49zcEJXy9R3r7gsBPBs43U66aqrh21DhEDXe9tpX5UZJZx4YZ7Iyk\nW1boCNRREcaJuF82IFSirR7XsPsL43AuLtbFtLXwiMofZlA+eKzGcH3HzNGq8KwA\nnwuEQxjhArqFGQ6PCrZd6HY0zRaI1XkGwC8AOznkbeo1DcL8yRCraMiEfsYh2kvy\n7ypsmkRBXOAubmSbAeKRXOWIH+R8w1VeE6GlRIpEUQKBgQC/+Yr1hSA/rJlECM81\nD8JYnJdpX9ngDMO8zZMHOQqXapfdnvXCJ4WS2pOhnGwhUEe11FEOcP4wKrprj0gz\nd2V5QTM4wtB9bWeFzJroeJHKUJ99YYuI6S7JVSuTv29pcD6tYI91VW4AW9f5Rpk0\n2SVwWU8r+X8ubirhnROqQSokHQKBgQC6lub18sTgBEsNrhSrMPvamDu3NPJlFLhs\ng6p5rBOkmld/tkHc4YElj19+vi+lwRuU0gEjjdrTXBcWdTn++BDMUeWpEnBx3LYL\nvnKOtX/RGfVA8gc4YeE0+zZLHeG57Ip52t4bzRgcVn6ZTs5t8vlJ7uvqqdR+QHJX\n6MI4aP1vkwKBgQCZSb7FcPlhHoZ7JrWdXuoGK3NTNrAYENkypsuh1tA4O2rsEYOW\n9kvYCSQcxXQp3ZqE+/WFHIA7IcMdI5m5Trr96SvnRNeJb5Rb6BZBThTLgTj4uqza\nM6eiJ5nWLePeQzwo4JNsUzy0mKGJb+/hnQoh/Y4URPJitqES6YPMTKBDmQKBgBRG\n5d6AfWiizs0zx8c60YPV21dzh4v4jnosbNBAJPpUU4HreojYcMJ2LDiHzoHC1I59\nq+YDOm6RqWilYKIWryylEcIn4NRe2eG41pYvny5IFeDy7FnyORka27GaE7eyvvGz\nGUQIK8CYnbVnXQORzgl8z2J3BkKaGlL3VnPu5OvFAoGAZ7kzeaJTFcGM1ZWiOrpo\nUhWBl4t3lqL9b7bKDL9lYisODUzQJ5t/4E+Sej66Ny5XeqxEFpMWDibmmPUMEmOG\nkWlgvDK1Z+1V9yDof0YDYyTqetu/e3KoX7jbvC8gUYW/+0AjnX8xVqB7i1Qz5PQi\n+gjqXZQNUF9dyzimFygX074=\n-----END PRIVATE KEY-----\n",
        "client_email": "credit-ai-sa@credit-ai-project-456311.iam.gserviceaccount.com",
        "client_id": "117752690884806967717",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/credit-ai-sa%40credit-ai-project-456311.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
        }
    )

def delete_gcs_folder(credentials, bucket_name, folder_name):
    storage_control_client = storage_control_v2.StorageControlClient(credentials=credentials)
    folder_path = storage_control_client.folder_path(project="_", bucket=bucket_name, folder=folder_name)
    request = storage_control_v2.DeleteFolderRequest(name=folder_path)
    storage_control_client.delete_folder(request=request)
    # return response.name


def create_gcs_folder(credentials, bucket_name, folder_id, recursive):
    storage_control_client = storage_control_v2.StorageControlClient(credentials=credentials)
    project_path = storage_control_client.common_project_path("_")
    bucket_path = f"{project_path}/buckets/{bucket_name}"

    # create folder called 'companies' where company files will be stored
    request = storage_control_v2.CreateFolderRequest(
        parent=bucket_path,
        folder_id=folder_id,
        recursive=recursive,
    )
    response = storage_control_client.create_folder(request=request)
    return response.name


# classify and summarise all files
def process_files_content_parts(bucket_name, blob, vertexai=True):
    gsutil_uri = 'gs://' + f'{bucket_name}/' + f'{blob.name}'
    # handle .msg files
    if gsutil_uri.endswith('.msg'):
        # download as bytes
        blob_dl_bytes = blob.download_as_bytes()
        # msg = extract_msg.Message(contents.decode("utf-8"))
        with extract_msg.Message(BytesIO(blob_dl_bytes)) as msg:
            # extract sender, date, subject, body into text string
            email_details = "\n".join([
                f"Sender: {msg.sender}",
                f"Sent on: {msg.date}",
                f"Subject: {msg.subject}",
                f"Email body: {msg.body}"
            ])
            if vertexai:
                attachment_content_ls = [
                    gm.Part.from_data(
                        data=attachment.data,
                        mime_type=attachment.mimetype
                    ) for attachment in msg.attachments
                ]
            else:
                attachment_content_ls = [
                    types.Part.from_bytes(
                        data=attachment.data,
                        mime_type=attachment.mimetype
                    ) for attachment in msg.attachments
                ]

        msg.close()
        if vertexai:
            file_content_parts = [gm.Part.from_text(text=email_details)] + attachment_content_ls
        else:
            file_content_parts = [types.Part.from_text(text=email_details)] + attachment_content_ls
    elif gsutil_uri.endswith('.xlsx'):
        # download as bytes
        blob_dl_bytes = blob.download_as_bytes()
        xls = pd.ExcelFile(BytesIO(blob_dl_bytes))
        file_content_parts = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(BytesIO(blob_dl_bytes), sheet_name=sheet_name)
            csv_str = df.to_csv(sep=',', index=False)
            prt = gm.Part.from_data(data=csv_str.encode(), mime_type="text/csv") if vertexai \
                else types.Part.from_bytes(data=csv_str.encode(), mime_type="text/csv")
            file_content_parts.append(prt)
        # contents = [Content(role="user", parts=content_parts)]
    elif gsutil_uri.endswith('.docx'):
        blob_dl_bytes = blob.download_as_bytes()
        doc = Document(BytesIO(blob_dl_bytes))
        txtstr = ""
        for par in doc.paragraphs:
            lines = par.text.split('\n')
            for line in lines:
                txtstr += line + '\n'

        file_content_parts = [gm.Part.from_text(text=txtstr)] if vertexai else [types.Part.from_text(text=txtstr)]
    elif (gsutil_uri.endswith('.pdf') or gsutil_uri.endswith('.py') or gsutil_uri.endswith('.txt')
          or gsutil_uri.endswith('.png') or gsutil_uri.endswith('.webp') or gsutil_uri.endswith('.jpeg')
          or gsutil_uri.endswith('.html') or gsutil_uri.endswith('.css') or gsutil_uri.endswith('.md')
          or gsutil_uri.endswith('.csv') or gsutil_uri.endswith('.xml')):
        # file = client.files.upload(file=file_path)
        # with open(file_path, 'rb') as f:
        #     pdf_bytes = f.read()
        # contents = [Part.from_uri(uri=gsutil_uri, mime_type=blob.content_type)]
        if vertexai:
            file_content_parts = [gm.Part.from_uri(uri=gsutil_uri, mime_type=blob.content_type)]
        else:
            file_content_parts = [types.Part.from_uri(file_uri=gsutil_uri, mime_type=blob.content_type)]
    else:
        raise TypeError(f"Unsupported file type: {gsutil_uri}")
    return file_content_parts


def change_company_name_gsc_to_vectordb(company_name):
    company_name_vector_db = company_name.replace(" ", "_").replace("(", "_").replace(")", "_")
    return company_name_vector_db

def get_company_list():
    client_wv = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    mt_col_companies = client_wv.collections.get("Companies")
    # list all tenant
    tenants = mt_col_companies.tenants.get()
    company_list = sorted([tenant.replace("_", " ") for tenant in tenants.keys() if tenant != "Cenomi_Centres"])
    client_wv.close()
    return company_list
