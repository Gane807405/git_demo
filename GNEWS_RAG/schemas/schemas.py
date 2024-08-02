from pydantic import BaseModel, ValidationError, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import Optional
class GnewsRequestModel(BaseModel):
    query: str
    rss_or_gnews: str = "rss"
    brand_name: str
    rss_link : Optional[str]=None
    "test2 from master"
    @field_validator('rss_or_gnews')
    def check_rss_or_gnews(cls, v):
        if v not in {"rss", "gnews"}:
            raise ValueError('rss_or_gnews must be either "rss" or "gnews"')
        return v
