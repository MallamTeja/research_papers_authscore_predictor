import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import signal

# Paths
DATASET_PATH = Path("dataset.json")
DATASET3_PATH = Path("dataset3.json")          # NEW: tier-3 dataset
EMBEDDING_DATASET_PATH = Path("embeddingdataset.json")
FLUSH_INTERVAL = 10

# Feature keys to check for non-zero (fixed to match your schema)
NONZERO_FEATURE_KEYS = (
    "manual_score",
    "impact_score",
    "rigor_score",
    "novelty_score",
)

# Global state
emb_model: SentenceTransformer | None = None
buffer: List[Dict[str, Any]] = []


def signal_handler(sig, frame):
    """Graceful flush on Ctrl+C."""
    print("\n[INTERRUPT] Flushing buffer before exit...")
    flush_buffer()
    sys.exit(0)


def load_model() -> None:
    """Load the SentenceTransformer model once into global state."""
    global emb_model
    if emb_model is not None:
        return
    print("Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")


def load_existing_pids() -> Set[str]:
    """Load set of already-embedded arxiv IDs to skip them."""
    if not EMBEDDING_DATASET_PATH.exists():
        return set()

    try:
        with open(EMBEDDING_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return set()

    # Defensive: handle non-list or missing fields
    if not isinstance(data, list):
        return set()

    return {str(item.get("arxiv_id")) for item in data if isinstance(item, dict)}


def record_has_nonzero_features(record: Dict[str, Any]) -> bool:
    """
    Return True only if all required feature keys exist and are non-zero (after safe casting).
    You can tweak the logic if you want 'any' non-zero instead of 'all'.
    """
    for key in NONZERO_FEATURE_KEYS:
        value = record.get(key, 0)

        # Handle None, strings, etc.
        if value is None:
            return False

        try:
            numeric_val = float(value)
        except (TypeError, ValueError):
            # If not castable to float, treat as invalid / zero-ish
            return False

        if numeric_val == 0:
            return False

    return True


def make_embedding_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw record → embedding record.
    Identical schema, but replaces 'text' with 'text_embedding'.
    """
    if emb_model is None:
        raise RuntimeError("Embedding model not loaded. Call load_model() first.")

    text = raw_record.get("text", "")
    if not isinstance(text, str):
        text = str(text)

    embedding = emb_model.encode(text).tolist()

    emb_record = raw_record.copy()
    emb_record["text_embedding"] = embedding
    if "text" in emb_record:
        del emb_record["text"]  # Replace text with embedding
    return emb_record


def flush_buffer() -> None:
    """Flush buffer to embeddingdataset.json by updating the whole list."""
    global buffer
    if not buffer:
        return

    existing: List[Dict[str, Any]] = []
    if EMBEDDING_DATASET_PATH.exists():
        try:
            with open(EMBEDDING_DATASET_PATH, "r", encoding="utf-8") as f:
                existing_json = json.load(f)
                if isinstance(existing_json, list):
                    existing = existing_json
        except (json.JSONDecodeError, FileNotFoundError):
            existing = []

    existing.extend(buffer)

    with open(EMBEDDING_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(
        f"Flushed {len(buffer)} records with embeddings. "
        f"Total in {EMBEDDING_DATASET_PATH}: {len(existing)}"
    )
    buffer.clear()


def load_records_from_path(path: Path) -> List[Dict[str, Any]]:
    """Load a dataset file if it exists, else return empty list."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return []
    if not isinstance(data, list):
        print(f"ERROR: {path} should contain a list of records.")
        return []
    return data


def main() -> None:
    global buffer

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Load model
    load_model()

    # Load from both datasets
    base_records  = load_records_from_path(DATASET_PATH)
    tier3_records = load_records_from_path(DATASET3_PATH)

    if not base_records and not tier3_records:
        print(f"ERROR: Neither {DATASET_PATH} nor {DATASET3_PATH} has any records to embed.")
        return

    all_records: List[Dict[str, Any]] = []
    all_records.extend(base_records)
    all_records.extend(tier3_records)

    print(
        f"Loaded {len(base_records)} records from {DATASET_PATH} and "
        f"{len(tier3_records)} records from {DATASET3_PATH} "
        f"(total={len(all_records)})."
    )

    # Load existing embeddings to skip
    existing_pids = load_existing_pids()
    print(f"Found {len(existing_pids)} existing embeddings. Skipping them.")

    # Filter new records by:
    # 1) not already embedded
    # 2) having all required non-zero features
    new_records: List[Dict[str, Any]] = []
    skipped_existing = 0
    skipped_zero_feature = 0

    for r in all_records:
        if not isinstance(r, dict):
            continue

        arxiv_id = str(r.get("arxiv_id"))
        if arxiv_id in existing_pids:
            skipped_existing += 1
            continue

        if not record_has_nonzero_features(r):
            skipped_zero_feature += 1
            continue

        new_records.append(r)

    print(
        f"Processing {len(new_records)} new records "
        f"(skipped {skipped_existing} existing, "
        f"{skipped_zero_feature} missing/zero-feature records)."
    )

    if not new_records:
        print("Nothing to embed: all eligible records are up to date.")
        return

    buffer = []
    for record in tqdm(new_records, desc="Embedding"):
        try:
            emb_record = make_embedding_record(record)
            buffer.append(emb_record)

            if len(buffer) >= FLUSH_INTERVAL:
                flush_buffer()

        except Exception as e:
            print(f"ERROR on {record.get('arxiv_id', 'unknown')}: {e}")
            continue  # Skip bad records

    # Final flush
    flush_buffer()
    print("Done. All eligible new records embedded.")


if __name__ == "__main__":
    main()
