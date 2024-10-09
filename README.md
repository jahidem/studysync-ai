## Prerequisite
Get an api key from:
  - https://aistudio.google.com/
  - https://qdrant.tech/

both of them are free. Now replace the `example.env` file with a `.env` file with valid api keys.

## installation & run (python 3.11.7)
```shell
python -m venv venv 
./venv/Scripts/activate
pip install -e .
studysync.exe
```
Open Swagger UI at http://localhost:8000/docs , all functionality can be tested there.
## Libraries used
- FastAPI
- Langchain
- Gemini API
- Qdrant

