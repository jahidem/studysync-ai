import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from studysync.routers.api import api
import uvicorn


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api, prefix="/api")

def run():
   uvicorn.run(app, host="0.0.0.0", port=8000)
  
if __name__ == "__main__":
   run()