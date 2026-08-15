#!/usr/bin/env python3
"""
Download AlphaFold predicted structures for all ProteinGym proteins.

Resolves UniProt entry names → accessions, downloads AlphaFold PDB files.

Usage: python3 download_structures.py
"""

import csv, os, sys, time, urllib.request, json, re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REF_CSV = os.path.join(DATA_DIR, "DMS_substitutions.csv")
PDB_CACHE = os.path.join(DATA_DIR, "pdb_cache")
MAPPING_FILE = os.path.join(PDB_CACHE, "uniprot_mapping.json")

ACCESSION_PATTERN = re.compile(
    r'^([OPQ][0-9][A-Z0-9]{3}[0-9]'
    r'|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}'
    r'|A0A[0-9A-Z]+)'
)


def resolve_accessions():
    """Build mapping from ProteinGym UniProt_ID → real UniProt accession."""
    # Check cache
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE) as f:
            return json.load(f)

    entries = {}
    with open(REF_CSV) as f:
        for row in csv.DictReader(f):
            uid = row["UniProt_ID"]
            entries[uid] = row

    mapping = {}
    need_lookup = {}

    for uid in entries:
        acc = uid.split("_")[0]
        if ACCESSION_PATTERN.match(acc):
            mapping[uid] = acc
        else:
            need_lookup[uid] = acc

    print(f"Direct accessions: {len(mapping)}")
    print(f"Need UniProt lookup: {len(need_lookup)}")

    # Resolve entry names via UniProt search
    for i, (uid, _) in enumerate(sorted(need_lookup.items())):
        url = f"https://rest.uniprot.org/uniprotkb/search?query=id:{uid}&format=json&fields=accession&size=1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ProteinGym-ASA/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            results = data.get("results", [])
            if results:
                mapping[uid] = results[0].get("primaryAccession", "?")
            else:
                print(f"  NOT FOUND: {uid}")
        except Exception as e:
            print(f"  ERROR: {uid} → {e}")
        time.sleep(0.3)
        if (i + 1) % 50 == 0:
            print(f"  ...resolved {i+1}/{len(need_lookup)}")

    print(f"Total resolved: {len(mapping)}/{len(entries)}")

    # Save mapping
    os.makedirs(PDB_CACHE, exist_ok=True)
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def download_structures(mapping):
    """Download AlphaFold PDB files for each accession."""
    # Get unique accessions
    unique_acc = {}
    for uid, acc in mapping.items():
        if acc not in unique_acc:
            unique_acc[acc] = uid

    print(f"\nUnique structures to download: {len(unique_acc)}")

    downloaded = 0
    skipped = 0
    failed = []

    for i, (acc, uid) in enumerate(sorted(unique_acc.items())):
        out_path = os.path.join(PDB_CACHE, f"{acc}.pdb")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            skipped += 1
            continue

        # AlphaFold URL
        url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ProteinGym-ASA/1.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            content = resp.read().decode("ascii", errors="replace")

            if content.startswith("<?xml") or "NoSuchKey" in content or len(content) < 500:
                failed.append((acc, uid, "not found or too small"))
                continue

            with open(out_path, "w") as f:
                f.write(content)

            downloaded += 1
            n_atoms = content.count("\nATOM")
            if (i + 1) % 25 == 0 or i < 5:
                print(f"  [{i+1}/{len(unique_acc)}] {acc} — {len(content)} bytes, {n_atoms} atoms")

            time.sleep(0.5)

        except Exception as e:
            failed.append((acc, uid, str(e)))

    print(f"\nDone: {downloaded} downloaded, {skipped} cached, {len(failed)} failed")

    if failed:
        fail_log = os.path.join(PDB_CACHE, "download_failures.txt")
        with open(fail_log, "w") as f:
            for acc, uid, err in failed:
                f.write(f"{acc}\t{uid}\t{err}\n")
        print(f"Failures logged to {fail_log}")
        for acc, uid, err in failed[:10]:
            print(f"  {acc} ({uid}): {err}")


def main():
    os.makedirs(PDB_CACHE, exist_ok=True)
    mapping = resolve_accessions()
    download_structures(mapping)

    # Print summary
    pdb_files = [f for f in os.listdir(PDB_CACHE) if f.endswith(".pdb")]
    print(f"\nTotal PDB files in cache: {len(pdb_files)}")


if __name__ == "__main__":
    main()
