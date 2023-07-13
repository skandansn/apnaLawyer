import random

from authentication import *
from apnaLawyer import langchain_query_processor, document_input_feeder
from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile
from typing import Union
import constants

app = FastAPI()

@app.get("/users/me")
async def read_items(current_user: Annotated[str, Depends(get_current_active_user)]):
    return {"user": current_user}

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
async def query(input_query: QueryInput, user: Annotated[str, Depends(get_current_active_user)]) ->  Union[QueryOutput, str]:
    processor_result = await langchain_query_processor(input_query, user)
    if isinstance(processor_result, str):
        return processor_result
    return QueryOutput(answer=processor_result[0], negation=processor_result[1])

@app.post("/upload-files")
async def create_upload_file(user: Annotated[str, Depends(get_current_active_user)], files: list[UploadFile]):
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
async def list_user_files(user: Annotated[str, Depends(get_current_active_user)]):
    os.makedirs("./storage/files/"+user.username, exist_ok=True)

    files = os.listdir("./storage/files/"+user.username)

    return {"Your uploaded files": [file for file in files]}