import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI
from studysync.routers.api import api
import uvicorn


app = FastAPI()

app.include_router(api, prefix="/api")

def run():
   uvicorn.run(app, host="0.0.0.0", port=8000)
  
if __name__ == "__main__":
   run()