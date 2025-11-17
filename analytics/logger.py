import pandas as pd
from datetime import datetime

df = pd.read_csv("messages.csv", encoding="utf-8",
                 names=["id", "author_id", "username", "display_name", 
                        "channel_id", "created_at", "length", 
                        "has_attachments", "status", "content"])

df["created_at"] = pd.to_datetime(df["created_at"])
df["has_attachments"] = df["has_attachments"].astype(bool)
df["was_edited"] = df["status"] == "edited"
df["deleted"] = df["status"] == "deleted"

df["hour"] = df["created_at"].dt.hour

user_stats = df.groupby("username").agg(
    total_messages=("id", "count"),
    non_text_ratio=("has_attachments", "mean"),
    edited_ratio=("was_edited", "mean"),
    off_hour_ratio=("hour", lambda x: ((x >= 6) & (x <= 11)).mean())
    )

for col in ["non_text_ratio", "edited_ratio", "off_hour_ratio"]:
    user_stats[col] = user_stats[col] * 100

user_stats = user_stats.sort_values("total_messages", ascending=False)
print(user_stats)
