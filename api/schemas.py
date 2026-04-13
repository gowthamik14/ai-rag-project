from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str

class AuthorisedKnowledgeRequest(BaseModel):
    model: str
    job_title: str
    job_cost: float
    make : str

class AuthorisedKnowledgeResponse(BaseModel):
    knowledge: str
    canbeAuthorised: bool
