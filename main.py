import random
import string

import dspy
import os
import json


class DocumentSignature(dspy.Signature):
    context: str = dspy.InputField()
    query: str = dspy.InputField()
    result: str = dspy.OutputField()

llm =dspy.LM(
    os.getenv("MODEL_NAME"), 
    api_key=os.getenv("OPEN_API_KEY"), 
    api_base=os.getenv("OPEN_API_URL")
)

dspy.configure(
    lm=llm
)

if __name__ == "__main__":
    rlm = dspy.RLM(DocumentSignature)
    text = ""
    with open("input.txt", "r") as f:
        text = f.read()
    
    result = rlm(
        context=text,
        query="What are the client responsibilities listed in the contract? List them as bullet points."
    )
    print(result.result)