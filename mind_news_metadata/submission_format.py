"""Pure helpers for the MIND CodaLab prediction.txt submission format."""
import os
import zipfile
from pathlib import Path


def format_impression_line(impression_id, rows):
    """Format one impression from ordered ``(candidate_position, probability)`` rows."""
    positions = [int(row[0]) for row in rows]
    if positions != list(range(len(rows))):
        raise ValueError(
            "candidate positions are not contiguous for impression {}".format(impression_id))
    rank_order = sorted(rows, key=lambda row: (-float(row[1]), int(row[0])))
    rank_by_position = {
        int(row[0]): rank + 1 for rank, row in enumerate(rank_order)
    }
    ranks = [rank_by_position[position] for position in positions]
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        raise ValueError("rank list is not a 1..N permutation")
    return "{} [{}]\n".format(impression_id, ",".join(map(str, ranks)))


def write_submission_zip(text_path, output):
    """Write a deterministic ZIP containing only root-level prediction.txt."""
    text_path, output = Path(text_path), Path(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    info = zipfile.ZipInfo("prediction.txt", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    try:
        with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(info, text_path.read_bytes())
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.namelist() != ["prediction.txt"]:
                raise ValueError("submission ZIP must contain only root-level prediction.txt")
            if archive.testzip() is not None:
                raise ValueError("submission ZIP failed CRC validation")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
