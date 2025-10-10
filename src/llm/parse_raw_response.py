import json
def parse_raw_response(reply):
    try:
        # First try direct JSON parsing
        reply = reply.strip()
        assertions = json.loads(reply)
        if isinstance(assertions, list):
            return assertions
        else:
            raise ValueError("Unexpected structure or count")
    except json.JSONDecodeError:
        # Try to extract a JSON list using regex
        try:
            match = re.search(r"```json(.*?)```", reply, re.DOTALL)
            if match:
                json_snippet = match.group(1)
                assertions = json.loads(json_snippet)
                if isinstance(assertions, list):
                    return assertions
                else:
                    raise ValueError("Extracted JSON structure incorrect")
        except Exception as e:
            raise ValueError("Failed to extract valid JSON from text") from e
    raise ValueError("Completely failed to parse or extract JSON")
