"""V5 dataset — CEDAR + SSBI combined, identity-disjoint, signer 7 locked."""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ai.dataset import WriterIndex, load_writer_index
from .config import V5Config

@dataclass
class WriterSamples:
    writer_id: str  # e.g. "C1" or "S5"
    genuine_paths: List[Path]
    forgery_paths: List[Path]
    is_ssbi: bool
    @property
    def n_genuine(self): return len(self.genuine_paths)
    @property
    def n_forgery(self): return len(self.forgery_paths)

def load_ssbi_index(root: Path) -> Dict[int, WriterIndex]:
    # root = .../data/sources/signatures
    genuine_labels = root / "genuine" / "labels.json"
    forged_labels = root / "forged" / "labels.json"
    with open(genuine_labels) as f: gj=json.load(f)
    with open(forged_labels) as f: fj=json.load(f)
    # Map image_id -> file_name
    gen_images = {img["id"]: img["file_name"] for img in gj["images"]}
    forg_images = {img["id"]: img["file_name"] for img in fj["images"]}
    # For each annotation, collect bbox crop? We need to provide per-writer sample paths.
    # The source sheets are large; we will crop signatures on the fly using bbox from labels.
    # For V5 we provide a SSBI index that stores (sheet_path, bbox) tuples, not just Path.
    # To keep compatible with WriterSamples, we will store a synthetic Path that encodes sheet+bbox via a custom object.
    # Instead, we will create a separate SSBI path list that train.py will handle with special preprocessing.
    # For now, return WriterIndex with dummy paths that encode person_id and annotation id.
    # We will store actual crops as temporary files? Instead, we will return a dict of person_id -> WriterIndex where originals are synthetic paths.
    # The train code will need to know how to load them.
    # Simplify: return the raw annotation data for later use.
    from collections import defaultdict
    gen_by_person = defaultdict(list)
    for ann in gj["annotations"]:
        pid=int(ann["attributes"]["person_id"])
        # store tuple (sheet file, bbox)
        sheet = gen_images[ann["image_id"]]
        sheet_path = root / "genuine" / "data" / sheet
        gen_by_person[pid].append((sheet_path, ann["bbox"]))
    forg_by_person = defaultdict(list)
    for ann in fj["annotations"]:
        pid=int(ann["attributes"]["person_id"])
        sheet = forg_images[ann["image_id"]]
        sheet_path = root / "forged" / "data" / sheet
        forg_by_person[pid].append((sheet_path, ann["bbox"]))

    # Build WriterIndex-like dict where we store the sheet+bbox as Path objects with special encoding
    # We will store the count and keep the lists for later
    index={}
    all_pids = set(list(gen_by_person.keys()) + list(forg_by_person.keys()))
    for pid in all_pids:
        # Create dummy WriterIndex with empty Path lists, but store counts
        # The actual paths will be handled via a separate SSBI store
        # For now, create WriterIndex with placeholder Paths that encode pid
        # We will use a custom dict outside this function
        index[pid]=WriterIndex(pid, [], [])
        # Attach extra attribute for SSBI
        index[pid].ssbi_genuine_crops = gen_by_person.get(pid, [])
        index[pid].ssbi_forgery_crops = forg_by_person.get(pid, [])
        # Also store counts
        index[pid].originals = [Path(f"ssbi_genuine_{pid}_{i}") for i in range(len(gen_by_person.get(pid,[])))]
        index[pid].forgeries = [Path(f"ssbi_forged_{pid}_{i}") for i in range(len(forg_by_person.get(pid,[])))]
    return index

def build_combined_index(cfg: V5Config):
    cedar_index = load_writer_index(cfg.cedar_root)
    ssbi_index = load_ssbi_index(cfg.ssbi_root)
    # Verify no overlap between cedar and ssbi ids (they are separate namespaces, but we use string keys)
    # For V5 we will keep them separate with prefixes
    combined={}
    for w in cfg.cedar_train + cfg.cedar_val + cfg.cedar_test:
        combined[f"C{w}"] = cedar_index[w]
    for w in cfg.ssbi_train + cfg.ssbi_val + cfg.ssbi_test + cfg.ssbi_locked:
        # ssbi_index keys are ints, but we need to map
        if w in ssbi_index:
            combined[f"S{w}"] = ssbi_index[w]
    return combined, cedar_index, ssbi_index

def assert_no_leakage(cfg: V5Config):
    train = set(cfg.cedar_train) | set(f"S{w}" for w in cfg.ssbi_train)
    val = set(cfg.cedar_val) | set(f"S{w}" for w in cfg.ssbi_val)
    test = set(cfg.cedar_test) | set(f"S{w}" for w in cfg.ssbi_test)
    locked = set(f"S{w}" for w in cfg.ssbi_locked)
    assert train.isdisjoint(val), "train∩val"
    assert train.isdisjoint(test), "train∩test"
    assert val.isdisjoint(test), "val∩test"
    assert locked.isdisjoint(train) and locked.isdisjoint(val), "locked leakage"
    assert 7 not in [int(x[1:]) for x in train if x.startswith("S")], "SSBI 7 in train"
    assert 7 not in [int(x[1:]) for x in val if x.startswith("S")], "SSBI 7 in val"
