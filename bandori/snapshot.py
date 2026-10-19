"""Freeze successful downloads without changing a running collector's files."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from collect_bandori_data import (Collector, atomic_json, make_templates,
                                 normalize_events, now_iso, select_events, summarize)


def snapshot_cache(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Snapshot output must be separate from the running download directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Snapshot directory must be new and empty")
    # The downloader writes state.json by atomic rename. This read observes one
    # complete committed version; later downloads cannot change this selection.
    state_bytes = (source / "state.json").read_bytes()
    state = json.loads(state_bytes)
    if not state:
        raise ValueError("No successful downloads available")
    output.mkdir(parents=True, exist_ok=True)
    for entry in state.values():
        raw = source / entry["path"]
        content = gzip.decompress(raw.read_bytes())
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Invalid source cache checksum: "+entry["path"])
        target = output / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(raw, target)
        except OSError:
            shutil.copy2(raw, target)
    atomic_json(output / "state.json", state)
    make_templates(output)
    for name in ("outages.csv", "schedule_revisions.csv", "verified_finals.csv"):
        path = source / "annotations" / name
        if path.exists():
            shutil.copy2(path, output / "annotations" / name)
    reader = Collector(output, offline=True)
    indexes = []
    for provider in ("bestdori", "hhwx"):
        item = reader.read_cache(provider+"/events")
        if item:
            indexes.append((provider, normalize_events(item[0], provider)))
    as_of = time.time()
    events, conflicts = select_events(indexes, as_of)
    if not events:
        raise ValueError("No CN schedule in the cache")
    selected = {}
    for key in state:
        parts = key.split("/")
        if len(parts) == 4 and parts[1] == "tracker":
            identity = (int(parts[2]), int(parts[3].removeprefix("tier_")))
            if identity not in selected or parts[0] == "bestdori":
                selected[identity] = {"key": key, "source": parts[0]}
    tiers = sorted({tier for _, tier in selected})
    if not tiers:
        raise ValueError("No cutoff histories available in cache")
    archives = reader.read_cache("bestdori/master/archives")
    stats = summarize(output, events, tiers, selected, archives[0] if archives else {},
                      as_of, 60, reader)
    timestamp = now_iso()
    atomic_json(output / "metadata_conflicts.json", conflicts)
    atomic_json(output / "manifest.json", {
        "snapshot": True, "source_cache": str(source), "collected_at_utc": timestamp,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(), "resources": state,
        "availability_assumption": "Historical publication times unknown unless provided per observation",
        "snapshot_policy": "Only successful, checksum-verified resources from one atomic cache-state read",
    })
    atomic_json(output / "summary.json", {"status": "partial_cache_snapshot", "created_at_utc": timestamp,
                                          "source_cache": str(source), **stats})
    atomic_json(output / "selection.json", {"event_ids": [e["event_id"] for e in events], "tiers": tiers})
    return stats
