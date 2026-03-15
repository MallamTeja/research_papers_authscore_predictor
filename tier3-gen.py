import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = Path("dataset.json")    # real tier1/2 papers
TIER3_PATH   = Path("dataset3.json")   # synthetic tier3 papers

GEMINI_MODEL   = "gemini-2.5-flash-lite"
TARGET_COUNT   = 100          # total tier3 papers desired in dataset3.json
MAX_RETRIES    = 3
RETRY_DELAY    = 70           # seconds between retries (free tier cooldown)
CALL_DELAY     = 10.0         # seconds between successful calls
FLUSH_EVERY    = 1            # flush after every generated record (like distillation)

# Conservative limits – we don't send any big texts, only prompts + topic
MAX_SUMMARY_TOKENS_HINT = 300
MAX_TEXT_TOKENS_HINT    = 1200

# ── Tier-3 generation rubric / prompt ────────────────────────────────────────

TIER3_RUBRIC = """
You are generating deliberately LOW-QUALITY ("tier-3") AI/ML research papers.

You will be given ONLY a TOPIC NAME. You must invent a fake, low-effort paper
about that topic, and output a SINGLE JSON OBJECT with a specific schema.

The paper must clearly look like:
- Very low research effort, minimal or fake experiments.
- Over-claims results without serious evidence.
- Poor structure, repetition, vague explanations.
- Awkward or basic vocabulary, sometimes misused.
- Sloppy punctuation and grammar (but still mostly readable).
- Very weak novelty, rigor, and impact.
- Buzzword stuffing, clichés, and hand-wavy statements.
- Obvious "tier-3" quality from a researcher's point of view.

SCORING RULES (VERY IMPORTANT):
- Fields: manual_score, novelty_score, rigor_score, impact_score.
- All scores are INTEGERS.
- manual_score MUST be between 0 and 40 inclusive.
- novelty_score, rigor_score, impact_score MUST be between 0 and 40 inclusive.
- manual_score MUST EQUAL (novelty_score + rigor_score + impact_score).
- For tier-3 papers, scores should usually be low (bottom half of the 0–40 range);
  do not generate good papers by mistake.

SCHEMA RULES (VERY IMPORTANT):
Return ONLY ONE JSON OBJECT, no markdown, no code fences. The object MUST have
all of these keys, with these types:

{
  "arxiv_id":            "<string>",
  "papernumber":         <int>,

  "manual_score":        <int>,
  "novelty_score":       <int>,
  "rigor_score":         <int>,
  "impact_score":        <int>,

  "title":               "<string>",
  "authorcount":         <int>,

  "summary":             "<string>",
  "summarylen":          <int>,

  "primarycategory":     "<string>",
  "categories":          ["<string>", "..."],

  "yearssincepublished": <float>,

  "text":                "<string>",
  "textlen":             <int>
}

ADDITIONAL RULES:
- "arxiv_id": invent a fake-looking arXiv ID string (e.g., "2503.12345v1").
- "papernumber": give any positive integer; it will be used only as an ID.
- "authorcount": small integer (1–5 is fine).
- "primarycategory": use a plausible arXiv-style category like "cs.LG", "cs.AI".
- "categories": list of 1–3 categories; include primarycategory in it.
- "yearssincepublished": a non-negative float (e.g., 0.0 to 15.0).
- "summary": short abstract (keep it under about 300 tokens).
- "text": low-quality full text; keep it under about 1200 tokens.
- "summarylen": length of "summary" (character count or token count, but be consistent).
- "textlen": length of "text" (same unit as summarylen).
- All strings must be valid JSON strings (escape quotes/newlines properly).
- DO NOT include any extra keys.
- DO NOT include markdown, comments, or any text outside the JSON object.

You MUST obey:
- manual_score in [0, 40]
- novelty_score in [0, 40]
- rigor_score in [0, 40]
- impact_score in [0, 40]
- manual_score = novelty_score + rigor_score + impact_score
- Low-quality, tier-3 paper style as described above.
""".strip()

# ── Gemini client ─────────────────────────────────────────────────────────────

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Dataset I/O ───────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of records")
    return data


def save_dataset(path: Path, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_existing_titles(dataset: List[Dict]) -> Set[str]:
    titles: Set[str] = set()
    for rec in dataset:
        title = rec.get("title") or rec.get("base_title") or ""
        if isinstance(title, str) and title.strip():
            titles.add(title.strip().lower())
    return titles


# ── Topic generation (local, no API) ─────────────────────────────────────────

BASE_TOPIC_SEEDS = [
    "Semi-supervised learning for noisy labels",
    "Federated learning for medical images",
    "Graph neural networks for traffic prediction",
    "Reinforcement learning for recommendation",
    "Contrastive learning for tabular data",
    "LLM-based code generation for robotics",
    "Meta-learning for few-shot classification",
    "Multimodal learning with audio and video",
    "AutoML for small datasets",
    "Time-series anomaly detection in IoT data",
    "Explainable deep learning for finance",
    "Data augmentation for low-resource NLP",
    "Self-supervised speech representation learning",
    "Neural architecture search for edge devices",
    "Causal discovery from observational data",
    "Adversarial robustness for vision models",
    "Knowledge distillation for tiny models",
    "Online learning for streaming data",
    "Domain adaptation for satellite imagery",
    "Generative models for synthetic tabular data",
]

def generate_topic_candidates() -> List[str]:
    """
    Generate a pool of possible topics locally, without calling the LLM.
    We can later filter them against dataset.json and dataset3.json.
    """
    topics: List[str] = []
    # Simple variations on base seeds
    for seed in BASE_TOPIC_SEEDS:
        topics.append(seed)
        topics.append(f"Toy study on {seed.lower()}")
        topics.append(f"Very simple approach to {seed.lower()}")
    # De-duplicate while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in topics:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(t.strip())
    return uniq


def filter_new_topics(
    candidates: List[str],
    existing_titles_dataset: Set[str],
    existing_titles_tier3: Set[str],
) -> List[str]:
    result: List[str] = []
    for t in candidates:
        k = t.strip().lower()
        if k and (k not in existing_titles_dataset) and (k not in existing_titles_tier3):
            result.append(t.strip())
    return result


# ── LLM JSON parsing / validation ─────────────────────────────────────────---

def parse_tier3_response(raw: str) -> Dict:
    """
    Parse and validate the JSON returned by Gemini for a tier-3 paper.
    We strip code fences like distillation.py, then extract the first {...} block.
    """
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw[:200]!r}")

    obj = json.loads(match.group())

    required_keys = {
        "arxiv_id",
        "papernumber",
        "manual_score",
        "novelty_score",
        "rigor_score",
        "impact_score",
        "title",
        "authorcount",
        "summary",
        "summarylen",
        "primarycategory",
        "categories",
        "yearssincepublished",
        "text",
        "textlen",
    }

    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # Basic type checks
    if not isinstance(obj["arxiv_id"], str):
        raise ValueError("arxiv_id must be string")
    if not isinstance(obj["papernumber"], int):
        raise ValueError("papernumber must be int")

    for k in ("manual_score", "novelty_score", "rigor_score", "impact_score"):
        if not isinstance(obj[k], int):
            raise ValueError(f"{k} must be int")

    if not isinstance(obj["title"], str):
        raise ValueError("title must be string")
    if not isinstance(obj["authorcount"], int):
        raise ValueError("authorcount must be int")
    if not isinstance(obj["summary"], str):
        raise ValueError("summary must be string")
    if not isinstance(obj["summarylen"], int):
        raise ValueError("summarylen must be int")
    if not isinstance(obj["primarycategory"], str):
        raise ValueError("primarycategory must be string")
    if not isinstance(obj["categories"], list):
        raise ValueError("categories must be list")
    if not isinstance(obj["yearssincepublished"], (int, float)):
        raise ValueError("yearssincepublished must be number")
    if not isinstance(obj["text"], str):
        raise ValueError("text must be string")
    if not isinstance(obj["textlen"], int):
        raise ValueError("textlen must be int")

    # Score constraints
    ms = obj["manual_score"]
    ns = obj["novelty_score"]
    rs = obj["rigor_score"]
    is_ = obj["impact_score"]

    if not (0 <= ms <= 40):
        raise ValueError(f"manual_score out of range: {ms}")
    for k, v in (("novelty_score", ns), ("rigor_score", rs), ("impact_score", is_)):
        if not (0 <= v <= 40):
            raise ValueError(f"{k} out of range: {v}")

    if ms != ns + rs + is_:
        raise ValueError(
            f"manual_score ({ms}) != novelty+rigor+impact ({ns}+{rs}+{is_})"
        )

    # Basic sanity on text/summary
    if len(obj["summary"].strip()) == 0:
        raise ValueError("summary is empty")
    if len(obj["text"].strip()) < 200:
        raise ValueError("text too short for a paper")

    return obj


def call_llm_for_topic(topic: str) -> Optional[Dict]:
    """
    Call Gemini to generate a single tier-3 paper for the given topic.
    """
    client = get_client()

    prompt = (
        f"{TIER3_RUBRIC}\n\n"
        f"TOPIC NAME: {topic}\n\n"
        f"Now generate ONE tier-3 paper JSON object for this topic, following ALL rules above."
    )

    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw_text = getattr(resp, "text", None)
            if not raw_text:
                raise ValueError("Empty response from model")
            parsed = parse_tier3_response(raw_text)
            print(
                f"  Generated tier-3 paper for topic '{topic}' "
                f"(manual_score={parsed['manual_score']}, "
                f"N={parsed['novelty_score']} R={parsed['rigor_score']} I={parsed['impact_score']})"
            )
            return parsed

        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] Parse/validation error: {e}")
        except Exception as e:
            last_err = e
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] API error: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    print(f"  [FAIL] All retries exhausted for topic '{topic}': {last_err}")
    return None


# ── Main ─────────────────────────────────────────────────────────────────────-

def main() -> None:
    # Pre-flight: validate API key before doing anything
    get_client()

    base_data = load_dataset(DATASET_PATH)
    tier3_data = load_dataset(TIER3_PATH)

    existing_titles_dataset = get_existing_titles(base_data)
    existing_titles_tier3   = get_existing_titles(tier3_data)

    already_have = len(tier3_data)
    if already_have >= TARGET_COUNT:
        print(f"Already have {already_have} tier-3 records in {TIER3_PATH}, nothing to do.")
        return

    remaining_needed = TARGET_COUNT - already_have

    print(
        f"{DATASET_PATH}: {len(base_data)} real papers\n"
        f"{TIER3_PATH}: {already_have} existing tier-3 papers\n"
        f"Need to generate: {remaining_needed}\n"
    )

    topic_candidates = generate_topic_candidates()
    topic_candidates = filter_new_topics(
        topic_candidates, existing_titles_dataset, existing_titles_tier3
    )

    if not topic_candidates:
        print("No available new topics after filtering against dataset and dataset3.")
        return

    # We may have more candidates than needed; trim
    topic_candidates = topic_candidates[:remaining_needed]

    generated = 0
    failed    = 0
    unsaved   = 0  # number of generated records not yet flushed

    try:
        for idx, topic in enumerate(topic_candidates, 1):
            if generated >= remaining_needed:
                break

            print(f"[{generated + 1}/{remaining_needed}] Topic: {topic}")

            result = call_llm_for_topic(topic)
            if result is None:
                failed += 1
                time.sleep(CALL_DELAY)
                continue

            # Attach topic as title if model didn't follow it, but usually it should
            # We keep model's title as-is, but also record the topic used.
            result["tier3_topic"] = topic

            tier3_data.append(result)
            generated += 1
            unsaved   += 1

            if unsaved >= FLUSH_EVERY:
                save_dataset(TIER3_PATH, tier3_data)
                print(
                    f"  [FLUSH] Saved — generated: {generated} | failed: {failed} | "
                    f"remaining topics in this run: {remaining_needed - generated}"
                )
                unsaved = 0

            time.sleep(CALL_DELAY)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted. Saving progress...")
        if unsaved > 0:
            save_dataset(TIER3_PATH, tier3_data)
        print(f"Saved. Generated {generated} tier-3 papers this run. Re-run to continue.")
        return

    except Exception as e:
        print(f"\n[ERROR] {e}. Saving progress...")
        if unsaved > 0:
            save_dataset(TIER3_PATH, tier3_data)
        print(f"Saved. Generated {generated} tier-3 papers this run. Re-run to continue.")
        return

    # Final flush (if FLUSH_EVERY > 1; here it's 1, but keep for safety)
    if unsaved > 0:
        save_dataset(TIER3_PATH, tier3_data)

    print(
        f"\nDone. Generated this run: {generated} | Failed: {failed} | "
        f"Total tier-3 now in {TIER3_PATH}: {len(tier3_data)}"
    )


if __name__ == "__main__":
    main()
