from openai import OpenAI, __version__
print("openai version:", __version__)
import os
import dotenv
dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import inspect
print("responses type:", type(client.responses))
print("responses.create:", client.responses.create)
print("responses.create qualname:", getattr(client.responses.create, "__qualname__", "<no qualname>"))
print("responses.create module:", getattr(client.responses.create, "__module__", "<no module>"))
print("responses.create signature:", inspect.signature(client.responses.create))

resp = client.responses.create(
    model="gpt-5-mini",
    input=[{
        "role": "user",
        "content": [{"type": "input_text", "text": "Return a JSON object with ok=true"}]
    }],
    reasoning={"effort": "minimal"},
    max_output_tokens=100,
    text={
        "format": {
            "type": "json_schema",
            "name": "Ok",                 # required here
            "schema": {                   # <-- move schema up to THIS level
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False
            },
            "strict": True                # <-- and strict sits here too
        }
    }
)

# Prefer parsed when you used a schema
parsed = getattr(resp, "output_parsed", None)
if parsed is not None:
    print("PARSED:", parsed)
else:
    # Fallback: collect any text from message items
    texts = []
    for item in resp.output or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    texts.append(part.text)
    print("TEXT:", "".join(texts))