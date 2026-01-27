import json
import re
from typing import cast

def parse_raw_response(reply : str) -> list[str]:
    try:
        # First try direct JSON parsing
        reply = reply.strip()
        data = json.loads(reply)

        # Ensure it is a list of strings
        if isinstance(data, list):
            return cast(list[str], data)
        else:
            raise ValueError("Extracted JSON structure incorrect, did not receive list of strings")
    except json.JSONDecodeError:
        # Try to extract a JSON list using regex
        try:
            match = re.search(r"```json(.*?)```", reply, re.DOTALL)
            if match:
                json_snippet = match.group(1)
                assertions = json.loads(json_snippet)
                if isinstance(assertions, list):
                    return cast(list[str], assertions)
                else:
                    raise ValueError("Extracted JSON structure incorrect, did not receive list of strings")
        except Exception as e:
            raise ValueError("Failed to extract valid JSON from text") from e
    raise ValueError("Completely failed to parse or extract JSON")
