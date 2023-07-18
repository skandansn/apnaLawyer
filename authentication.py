from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Annotated
from models import *
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv, find_dotenv
import os
from datetime import datetime, timedelta

load_dotenv(find_dotenv("local.env"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY =os.getenv('AUTH_SECRET')
REFRESH_TOKEN_SECRET_KEY =os.getenv('REFRESH_TOKEN_SECRET_KEY')
ALGORITHM = "HS256"

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "tier": 1,
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "tier": 0,
        "email": "alice@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def create_refresh_token(data: dict):
    refresh_expire = datetime.utcnow() + timedelta(days=30)
    refresh_token_data = {"sub": data["sub"]}
    refresh_token_data.update({"exp": refresh_expire})
    refresh_token = jwt.encode(refresh_token_data, REFRESH_TOKEN_SECRET_KEY, algorithm=ALGORITHM)
    return refresh_token

def create_access_token(data: dict, expires_delta: timedelta | None = None, should_refresh_token = False):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=300)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    refresh_token = None

    if should_refresh_token:
        refresh_token = create_refresh_token(data)

    return {"access_token": encoded_jwt, "refresh_token": refresh_token}


