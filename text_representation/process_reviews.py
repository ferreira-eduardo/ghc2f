"""Generate user and item text profiles from reviews with an LLM.

Usage: python process_reviews.py all_beauty --stage train [--category "All Beauty"]

Profiles are built only from interactions produced by preprocess/split_data.py:
  --stage train    : data/{dataset}/splits/train.parquet    -> data/{dataset}/profiles/train/
                     (training/tuning, evaluated on val)
  --stage trainval : data/{dataset}/splits/trainval.parquet -> data/{dataset}/profiles/trainval/
                     (final model, evaluated on test)
There is no test stage: test interactions are never used to build a profile.

Output: user_profile.parquet, item_profile.parquet, metadata.json
"""
import re, argparse, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_NAME = "Qwen/Qwen3-14B"            # or "Qwen/Qwen3-4B" / "Qwen/Qwen3-8B" for smaller GPUs
MAX_REVIEWS = 30                       # reviews per group sent to the LLM
MAX_CHARS = 1000                       # characters kept per review
BATCH_SIZE = 8                         # max prompts per batch
MAX_BATCH_TOKENS = 24000               # max padded tokens (prompt + new) per batch; fits a 48 GB GPU
MAX_NEW_TOKENS = 300
SEED = 42
torch.manual_seed(SEED)

GEN_KWARGS = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="dataset name, i.e. the folder data/{dataset}/splits")
parser.add_argument("--stage", required=True, choices=["train", "trainval"])
parser.add_argument("--category", default=None, help="optional item category mentioned in the prompts")
parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
args = parser.parse_args()

dataset = args.dataset
split_dir = args.data_root / dataset / "splits"
in_path = split_dir / f"{args.stage}.parquet"
out_dir = args.data_root / dataset / "profiles" / args.stage
category_clause = f" in the category {args.category}" if args.category else ""

USER_PROFILE_TEMPLATE = """Below are reviews written by one user about different items{category_clause}. \
Summarize this user's preferences as expressed in the reviews.

Step 1 - Filter: Some reviews may be uninformative. Completely ignore reviews whose content \
is only generic praise or complaints (e.g., "Great!", "Love it", "Terrible"), shipping, \
delivery, packaging, seller, customer service, store prices, items not yet used or given \
as gifts, or off-topic text. Base the profile ONLY on reviews that describe item aspects, \
the user's reasons for liking or disliking something, their needs, or their usage context.

Step 2 - Summarize, following these rules:
- Use ONLY information stated in the informative reviews. Do not infer demographics, \
personality, or preferences that are not expressed.
- Describe the user, not any single item. Prefer aspects that appear in several reviews.
- Include personal context only when it explains what the user needs from items \
(e.g., skin type, a condition, a hobby, experience level).
- Be specific and keep the user's own distinctive terms; avoid generic phrases that \
would describe any user (e.g., "values quality").
- If no review is informative, write "none" in every field.

Reviews:
{reviews}

Answer in exactly this format:
Likes: <aspects the user values, comma-separated>
Dislikes: <aspects the user criticizes, comma-separated>
Context: <usage situations, needs, or expertise mentioned>
Summary: <one or two sentences describing this user's preferences>"""

ITEM_PROFILE_TEMPLATE = """Below are reviews written by different users about one item{category_clause}. \
Summarize the characteristics of this item as described in the reviews.

Step 1 - Filter: Some reviews may be uninformative. Completely ignore reviews whose content \
is only generic praise or complaints (e.g., "Great!", "Love it", "Terrible"), shipping, \
delivery, packaging condition on arrival, seller, customer service, store prices, items \
not yet used, or off-topic text. Base the profile ONLY on reviews that describe the item \
or how it worked for the reviewer.

Step 2 - Summarize, following these rules:
- Use ONLY information stated in the informative reviews. Do not add product knowledge \
from elsewhere.
- Prefer aspects mentioned by several reviewers; mention disagreements briefly \
(e.g., "scent: liked by some, disliked by others").
- Be specific; avoid generic phrases that would describe any item.
- If no review is informative, write "none" in every field.

Reviews:
{reviews}

Answer in exactly this format:
Strengths: <aspects reviewers praise, comma-separated>
Weaknesses: <aspects reviewers criticize, comma-separated>
Suits: <users or usage situations the item fits, according to the reviews>
Summary: <one or two sentences describing the item>"""

USER_KEYS = ["Likes", "Dislikes", "Context", "Summary"]
ITEM_KEYS = ["Strengths", "Weaknesses", "Suits", "Summary"]

# ---------- data ----------
df = pd.read_parquet(in_path)

# ---------- leakage checks ----------
KEY = ["userId", "itemId", "timestamp"]

def assert_disjoint(df, split):
    held_out = pd.read_parquet(split_dir / f"{split}.parquet", columns=KEY).drop_duplicates()
    leaked = df[KEY].merge(held_out, on=KEY, how="inner")
    if len(leaked):
        raise SystemExit(
            f"LEAKAGE: {len(leaked)} (userId, itemId, timestamp) rows of {split}.parquet are "
            f"present in {in_path}. Refusing to build profiles.\n{leaked.head()}")

assert_disjoint(df, "test")
if args.stage == "train":
    assert_disjoint(df, "val")
n_input_rows = len(df)
print(f"leakage checks passed for stage '{args.stage}' ({in_path})")

df = df[["userId", "itemId", "review"]].copy()
df["userId"] = df["userId"].astype(str)
df["itemId"] = df["itemId"].astype(str)
df["review"] = df["review"].fillna("").astype(str).str.strip()
df = df[df.review != ""]

def group_reviews(col):
    """{id: [reviews]} with exact duplicates dropped and length capped."""
    groups = {}
    for key, revs in df.groupby(col, sort=False)["review"]:
        revs = list(dict.fromkeys(revs))
        groups[key] = [r[:MAX_CHARS] for r in revs[:MAX_REVIEWS]]
    return groups

user_groups = group_reviews("userId")
item_groups = group_reviews("itemId")
print(f"{dataset}: {len(df)} reviews, {len(user_groups)} users, {len(item_groups)} items")

# ---------- model ----------
tok = AutoTokenizer.from_pretrained(LLM_NAME)
tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype="auto", device_map="auto")
llm.eval()

def generate(prompts, desc):
    """Batched generation with thinking disabled, returned in the original order.
    Prompts are sorted longest first, so memory problems show up at the start
    instead of at the end of a run, and batches hold at most BATCH_SIZE prompts
    and MAX_BATCH_TOKENS padded tokens. A batch that still runs out of GPU
    memory is split in half and retried. Qwen recommends sampling
    (temp 0.7, top_p 0.8, top_k 20) in non-thinking mode rather than greedy."""
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]
    n_tok = [len(ids) + MAX_NEW_TOKENS for ids in tok(texts)["input_ids"]]
    order = sorted(range(len(prompts)), key=lambda k: -n_tok[k])
    batches, cur = [], []
    for k in order:  # the first prompt of a batch is its longest
        if cur and (len(cur) == BATCH_SIZE or (len(cur) + 1) * n_tok[cur[0]] > MAX_BATCH_TOKENS):
            batches.append(cur)
            cur = []
        cur.append(k)
    batches.append(cur)

    outputs = [None] * len(prompts)

    def run(idx):
        try:
            enc = tok([texts[k] for k in idx], return_tensors="pt", padding=True).to(llm.device)
            with torch.no_grad():
                gen = llm.generate(**enc, **GEN_KWARGS)
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            enc = gen = None
            torch.cuda.empty_cache()
            tqdm.write(f"OOM on a batch of {len(idx)} x {n_tok[idx[0]]} tokens, splitting it")
            run(idx[:len(idx) // 2])
            run(idx[len(idx) // 2:])
            return
        for k, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            text = tok.decode(g, skip_special_tokens=True)
            outputs[k] = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        pbar.update(len(idx))

    with tqdm(total=len(prompts), desc=desc) as pbar:
        for idx in batches:
            run(idx)
    return outputs

def fmt(revs):
    return "\n".join(f"{n}. {r}" for n, r in enumerate(revs, 1))

def match_field(text, k):
    return re.search(rf"^\s*\**{k}\**\s*:\s*(.+)$", text, flags=re.I | re.M)

def parse_fields(text, keys):
    out = {}
    for k in keys:
        m = match_field(text, k)
        out[k.lower()] = m.group(1).strip() if m else "none"
    return out

def parse_ok(text, keys):
    """True when every field was found in the output, so a "none" value is the
    model's answer and not a parse failure."""
    return all(match_field(text, k) for k in keys)

def build_profiles(groups, template, keys, id_col, desc):
    ids = list(groups)
    prompts = [template.format(category_clause=category_clause, reviews=fmt(groups[g])) for g in ids]
    outputs = generate(prompts, desc)
    return pd.DataFrame([{id_col: g, **parse_fields(t, keys), "raw_output": t, "parse_ok": parse_ok(t, keys)}
                         for g, t in zip(ids, outputs)])

# ---------- run ----------
out_dir.mkdir(parents=True, exist_ok=True)
user_df = build_profiles(user_groups, USER_PROFILE_TEMPLATE, USER_KEYS, "userId", "user profiles")
user_path = out_dir / "user_profile.parquet"
user_df.to_parquet(user_path, index=False)
print(f"saved {user_path} ({len(user_df)} rows, {(~user_df.parse_ok).sum()} parse failures)")

item_df = build_profiles(item_groups, ITEM_PROFILE_TEMPLATE, ITEM_KEYS, "itemId", "item profiles")
item_path = out_dir / "item_profile.parquet"
item_df.to_parquet(item_path, index=False)
print(f"saved {item_path} ({len(item_df)} rows, {(~item_df.parse_ok).sum()} parse failures)")

metadata = {
    "dataset": dataset,
    "stage": args.stage,
    "input_file": str(in_path.resolve()),
    "input_rows": int(n_input_rows),
    "reviews_used": int(len(df)),
    "category": args.category,
    "model": LLM_NAME,
    "decoding": {**GEN_KWARGS, "enable_thinking": False, "seed": SEED, "batch_size": BATCH_SIZE, "max_batch_tokens": MAX_BATCH_TOKENS},
    "MAX_REVIEWS": MAX_REVIEWS,
    "MAX_CHARS": MAX_CHARS,
    "user_profiles": {"rows": len(user_df), "parse_failures": int((~user_df.parse_ok).sum())},
    "item_profiles": {"rows": len(item_df), "parse_failures": int((~item_df.parse_ok).sum())},
    "date": datetime.now().isoformat(timespec="seconds"),
}
with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"saved {out_dir / 'metadata.json'}")
