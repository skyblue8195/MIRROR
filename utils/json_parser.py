"""Parse LLM output containing tuple of JSON objects separated by <split>."""
import json


def parse_json_tuple_with_split(text: str, expected_count: int = 2, default_first: str = None):
    """Parse text containing N JSON objects separated by <split> delimiter.

    Returns list of parsed JSON strings. Falls back to brace-matching if
    <split> delimiter is missing.
    """
    parts = text.split('<split>', 1)
    if len(parts) == expected_count:
        return [p.strip() for p in parts]

    # Fallback: extract JSON blocks via brace matching
    json_blocks = []
    i = 0
    while i < len(text) and len(json_blocks) < expected_count:
        start = text.find('{', i)
        if start == -1:
            break
        brace_count = 0
        in_string = False
        escaped = False
        for j in range(start, len(text)):
            c = text[j]
            if not in_string:
                if c == '"':
                    in_string = True
                    escaped = False
                elif c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start:j + 1]
                        try:
                            json.loads(candidate)
                            json_blocks.append(candidate)
                            i = j + 1
                            break
                        except json.JSONDecodeError:
                            pass
            else:
                if not escaped and c == '\\':
                    escaped = True
                elif escaped:
                    escaped = False
                elif c == '"':
                    in_string = False
        else:
            i = start + 1

    if len(json_blocks) >= expected_count:
        return json_blocks[:expected_count]
    elif len(json_blocks) == 1 and expected_count == 2:
        return [default_first or '', json_blocks[0]]
    elif len(json_blocks) == 0 and expected_count == 2 and default_first:
        return [default_first, text]
    return [text]
