from pydantic import BaseModel


class QueryInput(BaseModel):
    query: str
    negation: bool | None = False
    query_docs: bool | None = False
    kanoon: bool | None = False
    model: str | None = "gpt-3.5-turbo"


class QueryOutput(BaseModel):
    answer: str
    negation: str | None = ""


class User(BaseModel):
    username: str
    email: str
    tier: int | None = 0
    full_name: str | None = None
    disabled: bool | None = False

class UserInDB(User):
    hashed_password: str
    tier: int

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class RefreshToken(BaseModel):
    refresh_token: str

class RefreshTokenOutput(BaseModel):
    access_token: str

class TokenData(BaseModel):
    username: str | None = None