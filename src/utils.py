import json
import re
from typing import Optional


SYSTEM_PROMPT = (
    "You are a query decomposition engine. "
    "Given a complex multi-hop question, decompose it into atomic sub-queries. "
    "Each sub-query must be answerable by retrieving a SINGLE document chunk. "
    "Return ONLY a valid JSON array. "
    "Each element must have exactly these keys:\n"
    "  hop        (int)       — 1-indexed hop number\n"
    "  sub_query  (string)    — the atomic question for this hop\n"
    "  depends_on (list[int]) — hop numbers this query depends on (empty for hop 1)\n"
    "No explanation. No markdown. Only the JSON array."
)


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]


def build_prompt(question: str, tokenizer) -> str:
    messages = build_messages(question)
    prompt   = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def build_training_text(messages: list[dict], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _extract_json_block(text: str) -> Optional[str]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("[")
    end   = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return None

    return text[start : end + 1]


def parse_decomp_output(text: str) -> Optional[list[dict]]:
    block = _extract_json_block(text)
    if block is None:
        return None

    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list) or len(parsed) == 0:
        return None

    normalized = []
    for item in parsed:
        if not isinstance(item, dict):
            return None
        if "hop" not in item or "sub_query" not in item or "depends_on" not in item:
            return None
        normalized.append({
            "hop":        int(item["hop"]),
            "sub_query":  str(item["sub_query"]).strip(),
            "depends_on": [int(d) for d in item["depends_on"]],
        })

    return normalized


def validate_dep_graph(graph: list[dict]) -> tuple[bool, str]:
    if not graph:
        return False, "empty_graph"

    if len(graph) < 2:
        return False, "too_few_hops"

    hop_nums = {h["hop"] for h in graph}
    expected = set(range(1, len(graph) + 1))

    if hop_nums != expected:
        return False, f"hop_nums_wrong: got {sorted(hop_nums)}, expected {sorted(expected)}"

    for hop in graph:
        sq = hop.get("sub_query", "")
        if len(sq.split()) < 3:
            return False, f"sub_query_too_short: hop {hop['hop']}"

        for dep in hop.get("depends_on", []):
            if dep not in hop_nums:
                return False, f"invalid_depends_on: hop {hop['hop']} refs {dep}"

    return True, ""



def format_dep_graph(graph: list[dict]) -> str:
    if not graph:
        return "(empty)"
    lines = []
    for hop in graph:
        deps = f" [depends on: {hop['depends_on']}]" if hop["depends_on"] else " [independent]"
        lines.append(f"  Hop {hop['hop']}{deps}: {hop['sub_query']}")
    return "\n".join(lines)


def format_example_for_log(question: str, graph: list[dict], answer: str = "") -> str:
    out  = f"Q: {question}\n"
    out += format_dep_graph(graph)
    if answer:
        out += f"\nA: {answer}"
    return out


def get_token_length(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_messages_to_max_length(
    messages: list[dict],
    tokenizer,
    max_length: int,
) -> list[dict]:
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if get_token_length(full_text, tokenizer) <= max_length:
        return messages

    user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
    words    = messages[user_idx]["content"].split()

    while len(words) > 5:
        words = words[:-5]
        messages[user_idx] = {**messages[user_idx], "content": " ".join(words) + "..."}
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if get_token_length(full_text, tokenizer) <= max_length:
            break

    return messages

if __name__ == "__main__":
    valid_output = '[{"hop": 1, "sub_query": "Who directed Inception?", "depends_on": []}, {"hop": 2, "sub_query": "Where was Christopher Nolan born?", "depends_on": [1]}]'
    result = parse_decomp_output(valid_output)
    assert result is not None, "Failed to parse valid output"
    assert len(result) == 2
    assert result[0]["hop"] == 1
    assert result[1]["depends_on"] == [1]
    print("✓ parse_decomp_output — valid JSON")

    fenced = '```json\n' + valid_output + '\n```'
    result2 = parse_decomp_output(fenced)
    assert result2 is not None, "Failed to parse fenced output"
    print("✓ parse_decomp_output — markdown fences stripped")

    prose = 'Here is the decomposition:\n' + valid_output
    result3 = parse_decomp_output(prose)
    assert result3 is not None
    print("✓ parse_decomp_output — prose before JSON")

    assert parse_decomp_output("not json") is None
    assert parse_decomp_output("[]") is None           
    assert parse_decomp_output('{"not": "array"}') is None
    print("✓ parse_decomp_output — invalid inputs return None")

    ok, reason = validate_dep_graph(result)
    assert ok, f"Valid graph failed: {reason}"
    print(f"✓ validate_dep_graph — valid: {reason!r}")

    bad_graph = [{"hop": 1, "sub_query": "ok question here", "depends_on": [99]}]
    ok2, reason2 = validate_dep_graph(bad_graph)
    assert not ok2
    print(f"✓ validate_dep_graph — invalid: {reason2}")

    print("\n" + format_dep_graph(result))
    print("\n✓ All utils tests passed")