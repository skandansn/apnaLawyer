from datetime import datetime
import uuid
from typing import List
from pydantic import BaseModel, EmailStr, constr


class UserBaseSchema(BaseModel):
    name: str
    email: EmailStr
    photo: str | None = ""

    class Config:
        orm_mode = True


class CreateUserSchema(UserBaseSchema):
    password: constr(min_length=8)
    passwordConfirm: str
    role: str = 'free'
    verified: bool = True


class LoginUserSchema(BaseModel):
    email: EmailStr
    password: constr(min_length=8)


class UserResponse(UserBaseSchema):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class QueryInput(BaseModel):
    query: str
    negation: bool | None = False
    query_docs: bool | None = False
    kanoon: bool | None = False
    model: str | None = "gpt-3.5-turbo"


class QueryOutput(BaseModel):
    answer: str | None = None
    negation: str | None = None
    docsList: List | None = None


