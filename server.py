import random

from authentication import *
from routes import *
from apnaLawyer import langchain_query_processor, document_input_feeder
from fastapi import Request, Depends, FastAPI, HTTPException, status, File, UploadFile
from typing import Union
from fastapi.responses import JSONResponse
import constants
load_dotenv(find_dotenv("local.env"))

app = FastAPI()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    tier = get_route_tier(request.url.path)
    if tier == "public":
        response = await call_next(request)
        return response

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    SECRET_KEY = os.getenv('AUTH_SECRET')
    ALGORITHM = "HS256"
    authorization = request.headers.get('Authorization')
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]  # Extract the token value
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username)
            user = get_user(fake_users_db, username=token_data.username)
            if user is None:
                raise credentials_exception
            if user.disabled:
                return JSONResponse(status_code=401, content="Disabled user")
            request.state.token_payload = user
        except jwt.JWTError:
            return JSONResponse(status_code=401,content="Invalid token")
    else:
        return JSONResponse(status_code=401, content="Token missing")

    response = await call_next(request)
    return response


@app.get("/users/me")
async def user_info(request: Request):
    return {"user": request.state.token_payload}

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=300)
    tokens = create_access_token(
        data={"sub": user.username, "tier": 0}, expires_delta=access_token_expires, should_refresh_token=True
    )

    return {"access_token": tokens['access_token'], "refresh_token": tokens['refresh_token'], "token_type": "bearer"}

@app.post("/refresh-token")
def refresh_token(input_refresh_token: RefreshToken) -> RefreshTokenOutput:
    try:
        refresh_token_payload = jwt.decode(input_refresh_token.refresh_token, REFRESH_TOKEN_SECRET_KEY, algorithms=[ALGORITHM])
        new_access_token = create_access_token(refresh_token_payload)
        return new_access_token
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@app.get("/")
async def home():
    return {"message": "Welcome to ApnaLawyer!"}

@app.post("/query")
async def query(input_query: QueryInput, request: Request) ->  Union[QueryOutput, list]:
    processor_result = await langchain_query_processor(input_query, request.state.token_payload)
    if isinstance(processor_result, list):
        return processor_result
    return QueryOutput(answer=processor_result[0], negation=processor_result[1])

@app.post("/upload-files")
async def create_upload_file(request: Request, files: list[UploadFile]):
    user = request.state.token_payload
    if user.tier == 0:
        return constants.BAD_REQUEST_PERMISSION_DENIED

    os.makedirs("./storage/files/"+user.username, exist_ok=True)

    for file in files:
            file_path = os.path.join("./storage/files/"+user.username, file.filename)
            with open(file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)

    await document_input_feeder(user.username)

    return {"Uploaded filenames": [file.filename for file in files]}

@app.get("/list-files")
async def list_user_files(request: Request):
    user = request.state.token_payload
    os.makedirs("./storage/files/"+user.username, exist_ok=True)

    files = os.listdir("./storage/files/"+user.username)

    return {"Your uploaded files": [file for file in files]}