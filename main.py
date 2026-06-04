import random
import string

import dspy
import os
import json


class DocumentSignature(dspy.Signature):
    context: str = dspy.InputField()
    number: int = dspy.OutputField()

llm =dspy.LM(
    os.getenv("MODEL_NAME"), 
    api_key=os.getenv("OPEN_API_KEY"), 
    api_base=os.getenv("OPEN_API_URL")
)

dspy.configure(
    lm=llm
)

if __name__ == "__main__":
    secret_number = random.randint(100_000_000, 999_999_999)
    filler_lines = ["".join(random.choices(string.ascii_lowercase + " ", k=120)) for _ in range(50_000)]
    insert_at = random.randint(len(filler_lines) // 3, 2 * len(filler_lines) // 3)
    filler_lines.insert(insert_at, f"SECRET_NUMBER={secret_number}")
    haystack = "\n".join(filler_lines)
    print(f"Secret number inserted at line {insert_at}: {secret_number}")
    rlm = dspy.RLM(DocumentSignature)
    result = rlm(
        context="The context contains ~50k lines of random text with a single line "
    "matching the pattern SECRET_NUMBER=<digits>. Find and return ONLY the "
    f"numeric value.\n\n{haystack}"
    )
    print(result.number)