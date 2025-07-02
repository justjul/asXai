"""
asXai MCP Library
-----------------

Defines Pydantic models and utilities for the “Model Context Protocol” (MCP)
used to structure prompts and parse LLM JSON-like responses.

Components:
- parse_model_response: Extracts <think> segments and main content.
- parse_mcp_response: Safely locates and decodes JSON blocks in raw text.
- RobustKeyExtractor: Maps varied JSON keys to standardized fields.
- BaseModel subclasses (NotebookTitleMCP, ChatSummarizerMCP, etc.): 
  generate prompt templates and parse model outputs into structured data.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, TypedDict
import json
import re

import config
from asxai.logger import get_logger

logger = get_logger(__name__, level=config.LOG_LEVEL)


def parse_model_response(response: str) -> Dict[str, str]:
    """
    Splits model output into 'content' and optional 'think' segments
    demarcated by <think>...</think> tags.

    Args:
        response: Raw string from the LLM.

    Returns:
        Dict with keys:
          - 'content': visible content without think sections.
          - 'think': concatenated content inside <think> tags.
    """
    think_labels = [('<think>', '</think>')]

    for start_tag, end_tag in think_labels:
        if start_tag in response and end_tag in response:
            # Extract think segment
            pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
            think_match = re.search(pattern, response, flags=re.DOTALL)
            think = think_match.group(1).strip() if think_match else ""
            # Remove think segment from main content
            content = re.sub(pattern, "", response, flags=re.DOTALL).strip()
            return {"content": content, "think": think}
    # No think tags found
    return {"content": response.strip(), "think": ""}


def parse_mcp_response(text: str) -> Dict[str, Any]:
    """
    Locates the first JSON object in a text block and decodes it,
    with resilience to Python 'None' tokens.

    Args:
        text: Raw string potentially containing a JSON substring.

    Returns:
        Parsed dict if successful, else empty dict.
    """
    try:
        # Find JSON object boundaries
        start = text.index('{')
        end = text.rindex('}') + 1
        json_block = text[start:end]
        json_block = json_block.replace("None", "null")  \
            .replace("True", "true") \
            .replace("False", "false")
        return json.loads(json_block)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"JSON decoding error  in MCP response {text}: {e}")
        return {}


class RobustKeyExtractor:
    """
    Utility to normalize varied JSON key names into a fixed output schema.

    Example:
        key_map = {'title': ['title', 'topic'], ...}
        extractor = RobustKeyExtractor(key_map)
        extractor.extract(parsed_dict) -> {'title': value}
    """

    def __init__(self, key_map: Dict[str, List[str]]):
        self.key_map = key_map

    def extract(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts values for each target field from the record.

        Args:
            record: Parsed JSON dict from the model.

        Returns:
            Dict mapping each key_map target to its first matching value.
        """
        extracted = {}
        for target_field, candidates in self.key_map.items():
            value = self._extract_first(record, candidates)
            extracted[target_field] = value
        return extracted

    def _extract_first(self, record: Dict[str, Any], candidates: List[str]) -> Any:
        """
        Finds the first record key containing any of the candidate substrings.

        Args:
            record: JSON dict.
            candidates: List of substrings to match against record keys.

        Returns:
            Corresponding value or None if no match.
        """
        for key, value in record.items():
            if any(candidate.lower() in key.lower() for candidate in candidates):
                return value
        return None


class NotebookTitleMCP(BaseModel):
    """
    MCP model for generating notebook titles.
    Fields:
        title: Short 1-3 word title, no punctuation.
    """
    title: str = Field(
        description="A short, clear title summarizing the user query as a notebook topic. Should be 1-3 words. No periods.")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<QUERY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response: str) -> Dict[str, Any]:
        """
        Parses the LLM response into the NotebookTitleMCP schema.

        Args:
            response: Raw LLM output text.

        Returns:
            Dict with 'title' key.
        """
        msg = parse_model_response(response)
        res = parse_mcp_response(msg['content'])
        key_map = {
            'title': ['title', 'topic', 'heading'],
        }
        extractor = RobustKeyExtractor(key_map)
        return extractor.extract(res)


class ChatSummarizerMCP(BaseModel):
    """
    MCP model to summarize chat history into key topics.
    Fields:
        content: Summary string.
    """
    content: str = Field(
        description="A summary of the key topics discussed so far")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<HISTORY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt


class QueryParseMCP(BaseModel):
    """
    MCP model to parse structured query parameters.
    Fields:
        authorName, publicationDate_start/end, cleaned_query,
        peer_reviewed_only, preprint_only, venues
    """
    authorName: Optional[str] = Field(
        description="Author lastname(s) if specified"
    )
    publicationDate_start: Optional[str] = Field(
        description="Date to start searching from (formatted as YYYY-MM-DD even if there's just a year mentioned)"
    )
    publicationDate_end: Optional[str] = Field(
        description="Date to stop searching at (formatted as YYYY-MM-DD even if there's just a year mentioned)"
    )
    cleaned_query: Optional[str] = Field(
        description="The original query reformulated without author names or publication dates"
    )
    peer_reviewed_only: bool = Field(
        description="True or False; whether the user's is explicitely asking to return only peer-reviewed/published articles."
    )
    preprint_only: bool = Field(
        description="True or False; whether the user's is explicitely asking to return only preprint articles from a preprint repository."
    )
    venues: bool = Field(
        description="Venue the user is explicitely asking for, returned as a list of strings if applicable or an empty list otherwise. "
    )
    citationCount: bool = Field(
        description="Minimal number of citations articles should have."
    )

    @classmethod
    def generate_prompt(cls, instruct):
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<HISTORY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response):
        msg = parse_model_response(response)
        res = parse_mcp_response(msg['content'])
        key_map = {
            'authorName': ['authorName', 'author', 'name'],
            'publicationDate_start': ['publicationDate_start', 'start'],
            'publicationDate_end': ['publicationDate_end', 'end'],
            'peer_reviewed_only': ['peer', 'reviewed', 'published'],
            'preprint_only': ['preprint', 'repo'],
            'venues': ['venue', 'journal'],
            'citationCount': ['citationCount', 'citation', 'count'],
            'cleaned_query': ['cleaned_query', 'cleaned'],
        }
        extractor = RobustKeyExtractor(key_map)

        res = extractor.extract(res)
        res['cleaned_query'] = res.get(
            'cleaned_query') or res.get('query') or None
        return extractor.extract(res)


class ExpandQueryMCP(BaseModel):
    """
    MCP model to expand a query into sub-questions and flags.
    Fields:
        queries, scientific, ethical, cite_only, details, search_paperIds
    """
    queries: List[str] = Field(
        description="A list of 1-3 specific questions that collectively cover the user's query, returned as a list of strings."
        + "Each should be a clear, concise, and full sentence, suitable for retrieving relevant scientific documents.\n"
    )
    ethical: bool = Field(
        description="True or False; whether the user's question is ethically acceptable."
    )
    scientific: bool = Field(
        description="True or False; whether the user's question is 'scientific', i.e. in the scope of your expertise as a scientific assistant."
    )
    cite_only: bool = Field(
        description="True or False; whether the user's is explicitely asking to insert citation in the provided text."
    )
    details: bool = Field(
        description="True or False; whether the user's question is explicitely referring to a specific article mentioned in the conversation."
    )
    search_paperIds: bool = Field(
        description="Article IDs of the articles the user is explicitely referring to, returned as a list of strings if applicable or an empty list otherwise. "
    )

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<HISTORY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        return instruct.replace("<FIELDS>", "\n".join(fields))

    @classmethod
    def parse(cls, response):
        msg = parse_model_response(response)
        res = parse_mcp_response(msg['content'])
        key_map = {
            'queries': ['sub_queries', 'questions', 'queries'],
            'scientific': ['scientific', 'search', 'needed'],
            'cite_only': ['cite', 'citation', 'insert'],
            'ethical': ['ethical', 'valid', 'acceptable'],
            'details': ['details'],
            'search_paperIds': ['paperIds', 'search', 'paper'],
        }
        extractor = RobustKeyExtractor(key_map)
        return extractor.extract(res)


class KeywordsMCP(BaseModel):
    """
    MCP model to extract keywords and topics from a query.
    Fields:
        research_field, main_topics, key_concepts
    """
    research_field: str = Field(description="Relevant field of research among Computer science, Physics, Biology, or Medicine "
                                + "returned as a list of strings.")
    main_topics: List[str] = Field(
        description="3-5 topics of research related to the user's question, returned as a list of strings.")
    key_concepts: List[str] = Field(
        description="Specific concepts, keywords, or methods relevant to the search, returned as a list of strings.")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<HISTORY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        return instruct.replace("<FIELDS>", "\n".join(fields))

    @classmethod
    def parse(cls, response):
        msg = parse_model_response(response)
        res = parse_mcp_response(msg['content'])
        key_map = {
            'research_field': ['research_field', 'research', 'field', 'discipline', 'domain'],
            'main_topics': ['main_topics', 'main', 'topics'],
            'key_concepts': ['key_concepts', 'key', 'concept'],
        }
        extractor = RobustKeyExtractor(key_map)
        return extractor.extract(res)


class RelevantPaperMCP(BaseModel):
    """
    MCP model to filter candidate papers for relevance.
    Fields:
        relevant: boolean flag.
        paperIds: List of strings of IDs deemed relevant.
    """
    relevant: List[str] = Field(
        description="True or False; whether the provided articles are overall relevant to answer the user's question."
    )
    paperIds: List[str] = Field(
        description="A list of paper IDs from the provided candidates that are relevant to answer the user's question."
    )

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<HISTORY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response):
        msg = parse_model_response(response)
        print(f"Article Filter msg content {msg}")
        res = parse_mcp_response(msg['content'])
        print(f"Article Filter parsed messages {res}")

        key_map = {
            'relevant': ['relevant', 'valid'],
            'paperIds': ['paperId', 'paper', 'article', 'reference']
        }
        extractor = RobustKeyExtractor(key_map)

        return extractor.extract(res)


class SectionPlan(BaseModel):
    """
    Defines a section plan as part of a generation plan.
    Fields:
        title: Sub-question title for the section.
        scope: Description of section scope.
        paperIds: IDs of papers supporting this section.
    """
    title: str = Field(
        description="The title or sub-question to answer in this section")
    scope: str = Field(
        description="A short description of the topic covered in the section")
    paperIds: List[str] = Field(
        description="List of article Ids that should be used to generate this section")


class GenerationPlannerMCP(BaseModel):
    """
    MCP model to plan response structure.
    Fields:
        sections: List of SectionPlan entries.
        abstract: 2-3 sentence summary answering the query.
    """
    sections: List[SectionPlan]
    abstract: str = Field(
        description="A short summary of 2-3 sentences that answers the question")

    @classmethod
    def generate_prompt(cls, instruct: str) -> str:
        """
        Builds the prompt by listing expected fields with descriptions.

        Args:
            instruct: Template containing '<FIELDS>' and '<HISTORY>' placeholders.

        Returns:
            Populated prompt string.
        """
        fields = []
        for field_name, field_obj in cls.model_fields.items():
            description = field_obj.description or "No description"
            fields.append(f"- {field_name}: {description}")
        prompt = instruct.replace("<FIELDS>", "\n".join(fields))
        return prompt

    @classmethod
    def parse(cls, response):
        msg = parse_model_response(response)
        res = parse_mcp_response(msg['content'])

        abstract = res.get('abstract', [])
        sections = res.get('sections', [])

        key_map = {
            'title': ['title', 'topic'],
            'content': ['description', 'content', 'scope'],
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
