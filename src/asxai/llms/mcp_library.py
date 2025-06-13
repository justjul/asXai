from pydantic import BaseModel, field_validator, Field
from typing import Optional, List
from datetime import datetime
from dateutil.parser import parse as date_parse
import json

import config
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)


def parse_mcp_response(text: str):
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_block = text[start:end]
        print(json_block)
        json_block = json_block.replace("None", "null")
        dic = json.loads(json_block)
        print(dic)
        norm_dic = dic
        # default = datetime(1, 1, 1)  # defaults everything to Jan 1

        # def safe(val):
        #     try:
        #         parse(val, default=default).strftime("%Y-%m-%d")
        #         return True
        #     except Exception:
        #         return False

        # norm_dic = {'query': next((val for key, val in dic.items() if 'query' in key and val != 'null'), None),
        #             'cleaned_query': next((val for key, val in dic.items() if 'cleaned' in key and val != 'null'), None),
        #             'authorName': next((val for key, val in dic.items() if 'author' in key and val != 'null'), None),
        #             'publicationDate_start': next(
        #                 (parse(val, default=default).strftime("%Y-%m-%d")
        #                  for key, val in dic.items() if 'start' in key and safe(val)), None),
        #             'publicationDate_end': next(
        #                 (parse(val, default=default).strftime("%Y-%m-%d")
        #                  for key, val in dic.items() if 'end' in key and safe(val)), None)}

        return norm_dic
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"JSON decoding error: {e}")
        return {}


class NotebookTitleMCP(BaseModel):
    title: str = Field(
        description="A short, clear title summarizing the user query as a notebook topic. Should be 1-3 words. No periods.")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt


class ChatSummarizerMCP(BaseModel):
    content: str = Field(
        description="A summary of the key topics discussed so far")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt


class QueryParseMCP(BaseModel):
    query: Optional[str] = Field(description="The core search query")
    authorName: Optional[str] = Field(
        description="Author lastname(s) if specified")
    publicationDate_start: Optional[str] = Field(
        description="Date to start searching from (formatted as YYYY-MM-DD even if there's just a year mentioned)")
    publicationDate_end: Optional[str] = Field(
        description="Date to stop searching at (formatted as YYYY-MM-DD even if there's just a year mentioned)")
    cleaned_query: Optional[str] = Field(
        description="The original query reformulated without author names or publication dates")

    @classmethod
    def generate_prompt(cls, instruct):
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt


class ExpandQueryMCP(BaseModel):
    query: str = Field(
        description="The user's questions rephrased in a more descriptive and complete sentence, "
                    + "suitable for retrieving relevant scientific documents.")
    research_field: str = Field(description="Relevant fields of research among Computer science, Physics, Biology and Medecine "
                                + "returned as a list of strings.")
    main_topics: List[str] = Field(
        description="3-5 topics topics of research related to the user's question, returned as a list of strings.")
    key_concepts: List[str] = Field(
        description="Specific concepts, keywords, or methods relevant to the search, returned as a list of strings.")
    search_required: str = Field(
        description="'False' if the context of the conversation is sufficient to answer accurately the user's question, 'True' otherwise")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt


class SectionPlan(BaseModel):
    topic: str = Field(description="The topic or sub-question to answer")
    paperIds: List[str] = Field(
        description="List of article Ids that should be used to generate this section")


class GenerationPlannerMCP(BaseModel):
    sections: List[SectionPlan]

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt
