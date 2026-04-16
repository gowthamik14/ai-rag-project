from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(max_length=10_000)


class ChatResponse(BaseModel):
    reply: str


class AuthorisedKnowledgeRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    job_title: str = Field(min_length=1, max_length=500)
    job_cost: float = Field(ge=0, le=1_000_000)
    make: str = Field(min_length=1, max_length=200)


class AuthorisedKnowledgeResponse(BaseModel):
    knowledge: str
    canbeAuthorised: bool
    lastUpdatedBy: str = ""
    lastUpdatedDate: str = ""
