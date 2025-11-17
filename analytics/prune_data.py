from datetime import datetime
import json
import os
import re

from dotenv import load_dotenv

load_dotenv("./bot")
INPUT_JSONL = "messages.jsonl"
OUTPUT_JSONL = "pruned_dataset.jsonl"

MY_NAME = "seedhunter1"
USER_NAME = USER_NAME = os.environ["USER_NAME"]
MERGE_WINDOW = 8

ALLOWED_LIST = os.environ["ALLOWED_LIST"]

def clean_text(t):
    if not t:
        return ""
    
    while True:
        new_t = re.sub(r"\[[^\[\]]*\]", "", t)
        if new_t == t:
            break
        t = new_t

    t = re.sub(r"discord\.gg/\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t

def is_substantial_message(text):
    """Filters out commands and insubstantial messages to improve data quality"""
    if text.startswith('!'):
        return False

    # Short messages that are distinctive/reflective of your style should remain
    short_allowed = {ALLOWED_LIST}
    if len(text) < 15:
        if text.lower() not in short_allowed:
            return False

    if text.startswith('http'):
        return False
    
    return True

def load_messages(path):
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                msg = json.loads(line)
                msg["created_at"] = datetime.fromisoformat(msg["created_at"])
                cleaned = clean_text(msg["text"])
                if cleaned == "":
                    continue

                msg["text"] = cleaned
                msgs.append(msg)

    msgs.sort(key=lambda m: m["created_at"])

    merged = []
    current = None

    for msg in msgs:
        if current is None:
            current = msg
            continue

        same_author = (msg["author_name"] == current["author_name"])
        gap = (msg["created_at"] - current["created_at"]).total_seconds()

        if same_author and gap <= MERGE_WINDOW:
            current["text"] = current["text"] + " " + msg["text"]
            current["created_at"] = msg["created_at"]
        else:
            merged.append(current)
            current = msg

    if current:
        merged.append(current)

    deduped = []
    prev_text = None
    for m in merged:
        if m["text"] == prev_text:
            continue
        deduped.append(m)
        prev_text = m["text"]

    return deduped

def build_examples(messages):
    examples = []
    last_my_message_index = None

    for i, msg in enumerate(messages):
        is_me = (msg["author_name"] == MY_NAME)

        if not is_me:
            continue

        if not is_substantial_message(msg["text"]):
            last_my_message_index = i
            continue

        if last_my_message_index is None:
            context_msgs = messages[:i]
        else:
            context_msgs = messages[last_my_message_index + 1 : i]

        filtered = [
            m for m in context_msgs
            if (m["author_name"] != MY_NAME)
            and (m["text"].strip() != "")
            and not m["has_attachment"]
        ]

        filtered = filtered[-10:]

        example = {
            "messages": []
        }

        example["messages"].append({
            "role": "system",
            "content": f"You are {USER_NAME}, and you respond exactly in his natural Discord style and tone."
        })

        for ctx in filtered:
            example["messages"].append({
                "role": "user",
                "content": ctx["text"]
            })

        my_response = msg["text"].strip()
        if my_response == "":
            last_my_message_index = i
            continue

        example["messages"].append({
            "role": "assistant",
            "content": my_response
        })

        examples.append(example)
        last_my_message_index = i

    return examples

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    messages = load_messages(INPUT_JSONL)
    print(f"Loaded {len(messages)} messages.")

    examples = build_examples(messages)
    print(f"Built {len(examples)} training examples (your messages).")
    
    lengths = [len(ex["messages"][-1]["content"]) for ex in examples]
    print(f"Average response length: {sum(lengths) / len(lengths):.1f} chars")
    print(f"Responses > 50 chars: {sum(1 for l in lengths if l > 50)}")

    save_jsonl(OUTPUT_JSONL, examples)
    print(f"Wrote output to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()