"""Step 4, augmentation and generation: put the chunks from the search in the prompt. Stream the answer from Gemini."""
import sys

import httpx
from google import genai
from google.genai import errors, types

from .config import CHAT_MODEL, FALLBACK_MODEL
from .timing import tick

# Rules for the model: use only the reference material, give the source, and say "I do not know" if the answer is not there.
# These rules reduce hallucination.
SYSTEM_PROMPT = """You are the employee handbook assistant of the company.
Answer the question only from the "Reference material" that the user gives.
At the end of each sentence, give the source section.
If the reference material does not contain the answer, reply: "The handbook does not cover this. Contact HR."
Do not invent information."""


def build_prompt(question, context):
    return f"Reference material:\n{context}\n\nQuestion: {question}"


def generate(question, context):
    """Print the answer as a stream. If the main model fails, try the fallback model. If all models fail, exit the program."""
    for m in dict.fromkeys([CHAT_MODEL, FALLBACK_MODEL]):  # dict.fromkeys: if the two names are the same, try the model only once.
        tick(f"Send the request to {m}. Wait for the response (the model thinks first)…")
        try:
            # The client reads GEMINI_API_KEY. The with block keeps the client open until the stream ends.
            # After 60 s with no response, the call raises an error. It does not wait forever.
            with genai.Client(http_options=types.HttpOptions(timeout=60_000)) as client:
                for i, chunk in enumerate(client.models.generate_content_stream(
                    model=m,
                    contents=build_prompt(question, context),
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )):
                    if i == 0:
                        tick("Received the first part of the response (= thinking time + network delay)")
                        print(f"\nAnswer ({m}): ", end="")
                    print(chunk.text or "", end="", flush=True)
            print("\n")
            tick("Generation complete")
            return
        except (errors.APIError, httpx.HTTPError) as e:  # For example: 503 (service busy) or a timeout.
            tick(f"{m} failed → {type(e).__name__}: {str(e)[:150]}")
    sys.exit("All models failed. For 503 or a timeout, the free tier is busy. Try again later. For 400 or 403, check GEMINI_API_KEY.")
