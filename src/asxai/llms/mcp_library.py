from pydantic import BaseModel, field_validator, Field
from typing import Optional
from datetime import datetime
from dateutil.parser import parse as date_parse


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
