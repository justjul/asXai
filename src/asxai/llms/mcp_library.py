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
        logger.warning(f"JSON decoding error on {text}: {e}")
        return {}


class RobustKeyExtractor:
    def __init__(self, key_map):
        self.key_map = key_map

    def extract(self, record: dict):
        extracted = {}
        for target_field, candidates in self.key_map.items():
            value = self._extract_first(record, candidates)
            extracted[target_field] = value
        return extracted

    def _extract_first(self, record: dict, candidates: list):
        for key, value in record.items():
            if any(candidate in key.lower() for candidate in candidates):
                return value
        return None


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

    @classmethod
    def parse(cls, response):
        res = parse_mcp_response(response)
        key_map = {
            'title': ['title', 'topic', 'heading'],
        }
        extractor = RobustKeyExtractor(key_map)
        return extractor.extract(res)


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

    @classmethod
    def parse(cls, response):
        res = parse_mcp_response(response)
        key_map = {
            'query': ['query', 'question'],
            'authorName': ['authorName', 'author', 'name'],
            'publicationDate_start': ['publicationDate_start', 'start'],
            'publicationDate_end': ['publicationDate_end', 'end'],
            'cleaned_query': ['cleaned_query', 'cleaned'],
        }
        extractor = RobustKeyExtractor(key_map)

        res = extractor.extract(res)
        res['cleaned_query'] = res.get(
            'cleaned_query') or res.get('query') or None
        return extractor.extract(res)


class ExpandQueryMCP(BaseModel):
    query: str = Field(
        description="The user's questions rephrased in a more descriptive and complete sentence, "
                    + "suitable for retrieving relevant scientific documents.")
    research_field: str = Field(description="Relevant fields of research among Computer science, Physics, Biology and Medecine "
                                + "returned as a list of strings.")
    main_topics: List[str] = Field(
        description="3-5 topics of research related to the user's question, returned as a list of strings.")
    key_concepts: List[str] = Field(
        description="Specific concepts, keywords, or methods relevant to the search, returned as a list of strings.")
    search_needed: bool = Field(
        description="Whether additional article search is needed to answer this question (true or false)")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response):
        res = parse_mcp_response(response)
        key_map = {
            'query': ['query', 'question'],
            'research_field': ['research_field', 'research', 'field', 'discipline', 'domain'],
            'main_topics': ['main_topics', 'main', 'topics'],
            'key_concepts': ['key_concepts', 'key', 'concept'],
            'search_needed': ['search_needed', 'search'],
        }
        extractor = RobustKeyExtractor(key_map)
        return extractor.extract(res)


class SectionPlan(BaseModel):
    title: str = Field(
        description="The title or sub-question to answer in this section")
    scope: str = Field(
        description="A short description of the topic covered in the section")
    paperIds: List[str] = Field(
        description="List of article Ids that should be used to generate this section")


class GenerationPlannerMCP(BaseModel):
    sections: List[SectionPlan]
    abstract: str = Field(
        description="A short summary of 2-3 sentences that answers the question")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response):
        res = parse_mcp_response(response)

        abstract = res.get('abstract', [])
        sections = res.get('sections', [])

        key_map = {
            'title': ['title', 'topic'],
            'scope': ['description', 'content', 'scope'],
            'paperIds': ['paper', 'references', 'papers']
        }
        extractor = RobustKeyExtractor(key_map)
        sections = [extractor.extract(section) for section in sections]

        all_paperIds = [
            paperId for section in sections for paperId in section["paperIds"]]
        for section in sections:
            if "introduction" in section["title"].lower() or "tldr" in section["title"].lower():
                section["paperIds"] = all_paperIds

        return {"abstract": abstract, "sections": sections}


class SectionGenerationMCP(BaseModel):
    section_title: str = Field(
        description="The title of the section being answered")
    answer: str = Field(
        description="The generated answer for this section, with inline citations to articles using their paperId in square brackets")
    cited_papers: Optional[List[str]] = Field(
        description="List of paperIds actually cited in the answer")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt


class QuickReplyMCP(BaseModel):
    summary: str = Field(
        description="A concise scientific answer (max 2-3 paragraphs) that directly addresses the user query, using provided documents where possible. Inline citations use paperId in square brackets."
    )
    cited_papers: Optional[List[str]] = Field(
        description="List of paperIds actually cited in the answer"
    )

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response):
        res = parse_mcp_response(response)
        key_map = {
            'summary': ['summary', 'answer', 'response', 'reply'],
            'cited_papers': ['cited_papers', 'paperIds', 'papers', 'references'],
        }
        extractor = RobustKeyExtractor(key_map)
        return extractor.extract(res)
