from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


# Correct env variable
load_dotenv()  # or full absolute path


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=True
)

if __name__=="__main__":
    response = model.invoke("Hello")
    print(response)