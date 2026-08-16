import re
from typing import Optional, Dict, List
from fastapi import UploadFile, HTTPException


def extract_episode_number(filename: str) -> Optional[int]:

    # Try ep/eps first
    match = re.search(r'(?:^|[\s_\-])eps?[\s_]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback: last _number_ pattern
    matches = re.findall(r'_(\d+)(?:_|\.)', filename)
    if matches:
        return int(matches[-1])

    return None


def parse_and_validate_files(
        files: List[UploadFile],
        extensions: List[str]
) -> Dict[int, UploadFile]:

    filtered = [f for f in files if any(f.filename.endswith(ext) for ext in extensions)]

    unparsed = []
    episodes = {}
    duplicates = {}

    for f in filtered:
        ep_num = extract_episode_number(f.filename)

        if ep_num is None:
            unparsed.append(f.filename)
        elif ep_num in episodes:
            # Track duplicate
            if ep_num not in duplicates:
                duplicates[ep_num] = [episodes[ep_num].filename]
            duplicates[ep_num].append(f.filename)
        else:
            episodes[ep_num] = f

    # Reject if any file couldn't be parsed
    if unparsed:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract episode number from: {unparsed}"
        )

    # Reject if duplicates found
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate episode numbers found: {duplicates}"
        )

    return episodes


def validate_episode_pairs(
        set1: Dict[int, UploadFile],
        set2: Dict[int, UploadFile],
        name1: str = "files1",
        name2: str = "files2"
):
    """
    Validate that two sets have matching episode numbers.
    Raises HTTPException if mismatched.
    """
    eps1 = set(set1.keys())
    eps2 = set(set2.keys())

    missing_in_2 = eps1 - eps2
    missing_in_1 = eps2 - eps1

    if missing_in_2:
        raise HTTPException(
            status_code=400,
            detail=f"{name1} without matching {name2} for episodes: {sorted(missing_in_2)}"
        )
    if missing_in_1:
        raise HTTPException(
            status_code=400,
            detail=f"{name2} without matching {name1} for episodes: {sorted(missing_in_1)}"
        )