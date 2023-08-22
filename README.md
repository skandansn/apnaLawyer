# apnalawyer
A lawbot API which uses LLM for reasoning. Has multiple features including answering questions in documents, answering questions using LLMs like gpt3.5/4, using audio as an input for LLM questions instead of text, searching for Law cases/documents using third party APIs.

Tech:
Python / FastAPI framework
Langchain + Pinecone
Kanoon API integrations
Account management using JWT

How to run:
Create a .env file and add the secrets using the keys in samplelocal.env
Create a storage/files folder, which will be used for upload/list files now
Use docker compose to up the postgres sql db
Run the app with uvicorn app.main:app --reload
 
