"""Build a compact, raster-only Japanese glyph discrimination index.

The builder is intentionally the only part of this module that imports Pillow.
The :func:`load_index` API and ``GlyphDiscriminationIndex`` therefore work in
the main scoring environment with NumPy and the Python standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from mikan_font_probe.paths import ROOT
DEFAULT_RESOLUTIONS = (16, 24, 32, 48, 64)
DEFAULT_SOURCE_RESOLUTION = 128
LOW_DISTANCE_THRESHOLD = 0.12

STATUS_MISSING = 0
STATUS_EXACT_BITMAP = 1
STATUS_LOW_DISTANCE = 2
STATUS_DIFFERENT = 3
STATUS_SAME_FAMILY = 4
STATUS_UNKNOWN_RESOLUTION = 5
STATUS_NAMES = {
    STATUS_MISSING: "missing",
    STATUS_EXACT_BITMAP: "exact_bitmap",
    STATUS_LOW_DISTANCE: "low_distance",
    STATUS_DIFFERENT: "different",
    STATUS_SAME_FAMILY: "same_family",
    STATUS_UNKNOWN_RESOLUTION: "unknown_resolution",
}

# This is deliberately explicit rather than generated from a Unicode block:
# the index is meant for Japanese font choice, not as a claim of full Unicode
# coverage.  Precomposed dakuten/handakuten forms are included because a font
# can cover them independently from their decomposed sequences.
HIRAGANA = (
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞ"
    "ただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
    "まみむめもゃやゅゆょよらりるれろゎわゐゑをんゔ"
)
KATAKANA = (
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ"
    "マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヷヸヹヺ"
)
KANA_PUNCTUATION = "ー"

# A small, stable validation supplement.  The complete set from baseline,
# holdout, and active-selection files is added at build time below.
COMMON_KANJI = "日本語文字字体書体同異形見分区別共有近似高低大小上下左右日月火水木金土"


def _dedupe_characters(values: Iterable[str]) -> list[str]:
    characters: set[str] = set()
    for value in values:
        for char in value:
            if char.isspace() or unicodedata.category(char).startswith("C"):
                continue
            characters.add(char)
    return sorted(characters, key=ord)


def _read_character_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sample_characters() -> tuple[list[str], list[dict[str, str]]]:
    values = [HIRAGANA, KATAKANA, KANA_PUNCTUATION, COMMON_KANJI]
    sources: list[dict[str, str]] = []
    for filename in ("data/baseline_samples.json", "data/holdout_samples_v1.json"):
        path = ROOT / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        values.append("".join(glyph[2] for sample in data["samples"] for glyph in sample["glyphs"]))
        sources.append({"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    active = ROOT / "data/active-selection-v1/characters.txt"
    if active.exists():
        values.append(_read_character_file(active))
        sources.append({"path": str(active.relative_to(ROOT)), "sha256": hashlib.sha256(active.read_bytes()).hexdigest()})
    return _dedupe_characters(values), sources


def _resolve_extra(value: str) -> tuple[str, dict[str, str]]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.exists():
        content = _read_character_file(candidate)
        return content, {"path": str(candidate.relative_to(ROOT)), "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    return value, {"literal": value}


def _collect_characters(extra_values: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
    values, sources = _sample_characters()
    seen_sources = {json.dumps(source, ensure_ascii=False, sort_keys=True) for source in sources}
    for value in extra_values:
        content, source = _resolve_extra(value)
        values.append(content)
        source_key = json.dumps(source, ensure_ascii=False, sort_keys=True)
        if source_key not in seen_sources:
            sources.append(source)
            seen_sources.add(source_key)
    return _dedupe_characters(values), sources


def _shift_batch(images: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift an ``N x H x W`` ink batch without wrapping at the edge."""

    if dx == 0 and dy == 0:
        return images
    height, width = images.shape[-2:]
    shifted = np.zeros_like(images)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx) if dx >= 0 else width
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy) if dy >= 0 else height
    target_x0 = max(0, dx)
    target_x1 = target_x0 + (source_x1 - source_x0)
    target_y0 = max(0, dy)
    target_y1 = target_y0 + (source_y1 - source_y0)
    shifted[..., target_y0:target_y1, target_x0:target_x1] = images[..., source_y0:source_y1, source_x0:source_x1]
    return shifted


def _pairwise_distance(images: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Return symmetric F x F distances for one resolution and one glyph.

    Images are black-on-white uint8 rasters.  Distance is a bounded structural
    raster measure: 1 - soft ink IoU.  The
    second image is tried at the fixed center/four-neighbour shifts only; this
    absorbs one-pixel raster placement noise without searching a large
    alignment space.
    """

    ink = (255.0 - images.astype(np.float32)) / 255.0
    finite = np.full((images.shape[0], images.shape[0]), np.inf, dtype=np.float32)
    first = ink[:, None, :, :]
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = _shift_batch(ink, dx, dy)
        second = shifted[None, :, :, :]
        intersection = np.minimum(first, second).sum(axis=(-2, -1))
        union = np.maximum(first, second).sum(axis=(-2, -1))
        # For non-negative ink values, L1/union is exactly
        # 1 - intersection/union, so keep one scalar feature rather than
        # pretending these are independent signals.
        distance = 1.0 - intersection / np.maximum(union, 1e-6)
        finite = np.minimum(finite, distance.astype(np.float32))
    usable = coverage[:, None] & coverage[None, :]
    finite[~usable] = np.nan
    # The bounded shift set is symmetric, but explicitly enforcing symmetry
    # makes the persisted table and its lookup contract unambiguous.
    return np.minimum(finite, finite.T).astype(np.float32)


def _family_matrix(
    face_distances: np.ndarray,
    face_coverage: np.ndarray,
    families: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    family_count = len(families)
    resolution_count = face_distances.shape[2]
    character_count = face_distances.shape[3]
    distances = np.full((family_count, family_count, resolution_count, character_count), np.nan, dtype=np.float32)
    status = np.full(distances.shape, STATUS_MISSING, dtype=np.uint8)
    for family_a, info_a in enumerate(families):
        members_a = np.asarray(info_a["member_indices"], dtype=np.int64)
        covered_a = face_coverage[members_a]
        for family_b, info_b in enumerate(families):
            members_b = np.asarray(info_b["member_indices"], dtype=np.int64)
            covered_b = face_coverage[members_b]
            if family_a == family_b:
                available = covered_a.any(axis=0)
                distances[family_a, family_b, :, available] = 0.0
                status[family_a, family_b, :, available] = STATUS_SAME_FAMILY
                continue
            selected = face_distances[np.ix_(members_a, members_b)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                best = np.nanmin(selected, axis=(0, 1))
            available = covered_a.any(axis=0) & covered_b.any(axis=0)
            distances[family_a, family_b] = best
            distances[family_a, family_b, :, ~available] = np.nan
            status_view = status[family_a, family_b]
            status_view[available[None, :] & (best <= 0.0)] = STATUS_EXACT_BITMAP
            status_view[available[None, :] & (best > 0.0) & (best <= LOW_DISTANCE_THRESHOLD)] = STATUS_LOW_DISTANCE
            status_view[available[None, :] & (best > LOW_DISTANCE_THRESHOLD)] = STATUS_DIFFERENT
    return distances, status


def _render_glyphs(
    faces: list[dict[str, object]],
    characters: list[str],
    resolutions: Sequence[int],
    source_resolution: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Render visible-bbox-centered rasters; Pillow is imported lazily."""

    from PIL import Image, ImageDraw, ImageFont

    face_count = len(faces)
    character_count = len(characters)
    coverage = np.zeros((face_count, character_count), dtype=bool)
    glyphs: dict[int, np.ndarray] = {
        resolution: np.full((face_count, character_count, resolution, resolution), 255, dtype=np.uint8)
        for resolution in (*resolutions, source_resolution)
    }
    char_to_index = {char: index for index, char in enumerate(characters)}
    for face_index, face in enumerate(faces):
        supported = set(face["characters"])
        for char in supported:
            coverage[face_index, char_to_index[char]] = True
        path = ROOT / str(face["path"])
        for resolution in glyphs:
            renderer = ImageFont.truetype(str(path), resolution)
            for char in supported:
                left, top, right, bottom = renderer.getbbox(char)
                width, height = right - left, bottom - top
                image = Image.new("L", (resolution, resolution), 255)
                x = round((resolution - width) / 2 - left)
                y = round((resolution - height) / 2 - top)
                ImageDraw.Draw(image).text((x, y), char, font=renderer, fill=0)
                glyphs[resolution][face_index, char_to_index[char]] = np.asarray(image, dtype=np.uint8)
    return glyphs, coverage


def _select_evidence(
    faces: list[dict[str, object]],
    families: list[dict[str, object]],
    characters: list[str],
    coverage: np.ndarray,
    glyphs: dict[int, np.ndarray],
    face_distances: np.ndarray,
    family_distances: np.ndarray,
    family_status: np.ndarray,
) -> list[dict[str, object]]:
    """Choose a few computed examples for visual checking, not a test set."""

    kana_set = set(HIRAGANA + KATAKANA + KANA_PUNCTUATION)
    char_order = [char for char in characters if char in kana_set]
    # The evidence image is intentionally kana-first: this artifact is for
    # human raster sanity checks, not a representative sample or a benchmark.
    # Fall back to the full character set only if no kana is covered.
    if not char_order:
        char_order = list(characters)
    char_indices = [characters.index(char) for char in char_order]
    rows: list[dict[str, object]] = []
    for face_index, face in enumerate(faces):
        if "な" in face["characters"]:
            char_index = characters.index("な")
            if coverage[face_index, char_index]:
                rows.append({
                    "kind": "same_face_zero",
                    "char": "な",
                    "resolution": 32,
                    "face_a": face_index,
                    "face_b": face_index,
                    "distance": 0.0,
                    "status": "exact_bitmap",
                })
                break

    def best_cross(status_code: int | None, maximum: bool) -> dict[str, object] | None:
        candidates: list[tuple[float, int, int, int, int]] = []
        resolution_index = 0 if not maximum else len(DEFAULT_RESOLUTIONS) - 1
        for family_a in range(len(families)):
            for family_b in range(family_a + 1, len(families)):
                for char_index in char_indices:
                    status_value = int(family_status[family_a, family_b, resolution_index, char_index])
                    if status_code is not None and status_value != status_code:
                        continue
                    distance = float(family_distances[family_a, family_b, resolution_index, char_index])
                    if not math.isfinite(distance):
                        continue
                    members_a = families[family_a]["member_indices"]
                    members_b = families[family_b]["member_indices"]
                    pair = face_distances[np.ix_(members_a, members_b)][:, :, resolution_index, char_index]
                    pair_index = np.unravel_index(np.nanargmin(pair) if not maximum else np.nanargmax(pair), pair.shape)
                    face_a = int(members_a[pair_index[0]])
                    face_b = int(members_b[pair_index[1]])
                    candidates.append((distance, family_a, family_b, char_index, face_a * len(faces) + face_b))
        if not candidates:
            return None
        selected = (max if maximum else min)(candidates, key=lambda item: item[0])
        distance, family_a, family_b, char_index, packed = selected
        face_a, face_b = divmod(packed, len(faces))
        return {
            "kind": "cross_family_high_distance" if maximum else "cross_family_low_distance",
            "char": characters[char_index],
            "resolution": DEFAULT_RESOLUTIONS[resolution_index],
            "face_a": face_a,
            "face_b": face_b,
            "family_a": families[family_a]["key"],
            "family_b": families[family_b]["key"],
            "distance": distance,
            "status": STATUS_NAMES[int(family_status[family_a, family_b, resolution_index, char_index])],
        }

    low = best_cross(STATUS_LOW_DISTANCE, maximum=False)
    if low is None:
        low = best_cross(None, maximum=False)
    if low is not None:
        rows.append(low)
    high = best_cross(STATUS_DIFFERENT, maximum=True)
    if high is None:
        high = best_cross(None, maximum=True)
    if high is not None:
        rows.append(high)
    return rows


def _write_evidence(
    output: Path,
    rows: list[dict[str, object]],
    faces: list[dict[str, object]],
    glyphs: dict[int, np.ndarray],
    characters: list[str],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    evidence_dir = output / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for row in rows:
        row["face_id_a"] = faces[int(row["face_a"])] ["id"]
        row["face_id_b"] = faces[int(row["face_b"])] ["id"]
        row["face_path_a"] = faces[int(row["face_a"])] ["path"]
        row["face_path_b"] = faces[int(row["face_b"])] ["path"]

    cell_size = 192
    label_height = 36
    canvas = Image.new("RGB", (cell_size * 3, (cell_size + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y0 = row_index * (cell_size + label_height)
        resolution = int(row["resolution"])
        char_index = characters.index(str(row["char"]))
        left = Image.fromarray(glyphs[resolution][int(row["face_a"]), char_index], mode="L").convert("RGB")
        right = Image.fromarray(glyphs[resolution][int(row["face_b"]), char_index], mode="L").convert("RGB")
        diff = Image.fromarray(np.abs(np.asarray(left.convert("L"), dtype=np.int16) - np.asarray(right.convert("L"), dtype=np.int16)).astype(np.uint8), mode="L").convert("RGB")
        scale = max(1, cell_size // resolution)
        target = resolution * scale
        x = (cell_size - target) // 2
        for column, image in enumerate((left, right, diff)):
            image = image.resize((target, target), Image.Resampling.NEAREST)
            canvas.paste(image, (column * cell_size + x, y0 + label_height + (cell_size - target) // 2))
            draw.rectangle((column * cell_size, y0 + label_height, (column + 1) * cell_size - 1, y0 + label_height + cell_size - 1), outline="#999999")
        title = f"{row['kind']} U+{ord(str(row['char'])):04X} r{resolution} d={float(row['distance']):.4f}"
        draw.text((4, y0 + 4), title, fill="black", font=label_font)
        draw.text((cell_size + 4, y0 + 4), "A", fill="black", font=label_font)
        draw.text((cell_size * 2 + 4, y0 + 4), "B / abs diff", fill="black", font=label_font)
    canvas.save(evidence_dir / "comparison-examples.png")
    (evidence_dir / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_index(output: Path, extra_values: Sequence[str] = (), resolutions: Sequence[int] = DEFAULT_RESOLUTIONS, source_resolution: int = DEFAULT_SOURCE_RESOLUTION) -> dict[str, object]:
    """Render faces and write ``index.json`` plus compact NumPy arrays."""

    from PIL import __version__ as pillow_version, features

    from mikan_font_probe.build_font_templates import font_metadata
    from mikan_font_probe.font_family_policy import family_policy

    characters, character_sources = _collect_characters(extra_values)
    resolution_list = tuple(dict.fromkeys(int(resolution) for resolution in resolutions))
    if not resolution_list or any(resolution <= 0 for resolution in resolution_list):
        raise ValueError("resolutions must contain positive integers")
    if source_resolution in resolution_list:
        raise ValueError("source resolution must be separate from measured resolutions")

    output.mkdir(parents=True, exist_ok=True)
    faces: list[dict[str, object]] = []
    for path in sorted((ROOT / "font_sample").rglob("*")):
        if path.suffix.lower() not in (".otf", ".ttf"):
            continue
        metadata = font_metadata(path, characters)
        policy = family_policy(metadata["postscript"], path.parent.name)
        faces.append({
            "id": metadata["sha256"][:16],
            "path": str(path.relative_to(ROOT)),
            "postscript": metadata["postscript"],
            "family": metadata["family"],
            "style": metadata["style"],
            "weight": metadata["weight"],
            "family_key": policy["key"],
            "family_name": policy["name"],
            "answer": policy["answer"],
            "characters": metadata["characters"],
            "vertical_tables": metadata["vertical_tables"],
        })
    if len(faces) != 23:
        raise RuntimeError(f"expected 23 font files, found {len(faces)}")

    family_keys = sorted({str(face["family_key"]) for face in faces})
    families: list[dict[str, object]] = []
    for family_key in family_keys:
        members = [index for index, face in enumerate(faces) if face["family_key"] == family_key]
        first = faces[members[0]]
        families.append({
            "key": family_key,
            "name": first["family_name"],
            "answer": first["answer"],
            "member_indices": members,
            "members": [faces[index]["id"] for index in members],
        })
    if len(families) != 14:
        raise RuntimeError(f"expected 14 family definitions, found {len(families)}")

    glyphs, coverage = _render_glyphs(faces, characters, resolution_list, source_resolution)
    face_distances = np.full((len(faces), len(faces), len(resolution_list), len(characters)), np.nan, dtype=np.float32)
    for resolution_index, resolution in enumerate(resolution_list):
        for character_index in range(len(characters)):
            face_distances[:, :, resolution_index, character_index] = _pairwise_distance(
                glyphs[resolution][:, character_index], coverage[:, character_index]
            )
    family_distances, family_status = _family_matrix(face_distances, coverage, families)
    for family_index, family in enumerate(families):
        family["member_indices"] = [int(index) for index in family["member_indices"]]

    status_counts: dict[str, dict[str, int]] = {}
    for resolution_index, resolution in enumerate(resolution_list):
        counts = {name: 0 for name in STATUS_NAMES.values()}
        for family_a in range(len(families)):
            for family_b in range(family_a + 1, len(families)):
                for code in family_status[family_a, family_b, resolution_index].flat:
                    counts[STATUS_NAMES[int(code)]] += 1
        status_counts[str(resolution)] = counts

    npz_path = output / "glyphs.npz"
    arrays: dict[str, np.ndarray] = {
        f"glyph_r{resolution}": glyphs[resolution] for resolution in resolution_list
    }
    arrays[f"glyph_r{source_resolution}"] = glyphs[source_resolution]
    arrays.update({
        "coverage": coverage,
        "face_distance": face_distances,
        "family_distance": family_distances,
        "family_status": family_status,
        "char_codepoints": np.asarray([ord(char) for char in characters], dtype=np.int32),
    })
    np.savez_compressed(npz_path, **arrays)

    index: dict[str, object] = {
        "schema_version": 1,
        "purpose": "raster glyph discrimination for Japanese font-family selection; not a probability or vector-outline identity table",
        "character_sources": character_sources,
        "characters": characters,
        "resolutions": list(resolution_list),
        "source_resolution": source_resolution,
        "low_distance_threshold": LOW_DISTANCE_THRESHOLD,
        "status_codes": {name: code for code, name in STATUS_NAMES.items()},
        "rendering": {
            "engine": "Pillow FreeType",
            "pillow": pillow_version,
            "freetype": features.version("freetype2"),
            "cell": "resolution x resolution grayscale uint8, white background",
            "placement": "visible glyph bbox centered in fixed cell; no fallback; isolated upright glyph",
            "distance": "minimum of center and four-neighbour translations; 1 - soft ink IoU (L1/union is algebraically identical, not an independent feature)",
            "vector_identity": "not measured; no outline comparison",
        },
        "fonts": faces,
        "families": families,
        "arrays": {"path": npz_path.name, "format": "NumPy NPZ, allow_pickle=False"},
        "summary": {
            "font_count": len(faces),
            "family_count": len(families),
            "character_count": len(characters),
            "coverage_count": int(coverage.sum()),
            "status_counts_off_diagonal_family_pairs": status_counts,
        },
    }
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = _select_evidence(faces, families, characters, coverage, glyphs, face_distances, family_distances, family_status)
    _write_evidence(output, rows, faces, glyphs, characters)
    return index


class GlyphDiscriminationIndex:
    """Read-only NumPy/JSON loader for the generated discrimination table."""

    def __init__(self, index_path: Path, metadata: dict[str, object]):
        self.index_path = index_path
        self.root = index_path.parent
        self.metadata = metadata
        self.characters = tuple(str(char) for char in metadata["characters"])
        self.character_index = {char: index for index, char in enumerate(self.characters)}
        self.resolutions = tuple(int(resolution) for resolution in metadata["resolutions"])
        self.resolution_index = {resolution: index for index, resolution in enumerate(self.resolutions)}
        self.source_resolution = int(metadata["source_resolution"])
        self.faces = list(metadata["fonts"])
        self.face_index = {str(face["id"]): index for index, face in enumerate(self.faces)}
        self.families = list(metadata["families"])
        self.family_index = {str(family["key"]): index for index, family in enumerate(self.families)}
        self._arrays = np.load(self.root / str(metadata["arrays"]["path"]), allow_pickle=False)
        # NpzFile key access can decompress a member on every lookup.  Keep
        # hot scoring arrays as ndarray references for repeated pair queries;
        # glyph rasters are cached lazily because callers may only need one
        # resolution or a small subset of faces.
        self.coverage = self._arrays["coverage"]
        self.face_distance_table = self._arrays["face_distance"]
        self.family_distance_table = self._arrays["family_distance"]
        self.family_status_table = self._arrays["family_status"]
        self._glyph_cache: dict[str, np.ndarray] = {}

    def _resolution_index(self, resolution: int) -> int | None:
        requested = int(resolution)
        if requested <= 0:
            raise ValueError("resolution must be positive")
        if requested < min(self.resolutions):
            return None
        exact = self.resolution_index.get(requested)
        if exact is not None:
            return exact
        lower = [value for value in self.resolutions if value <= requested]
        # At or above the smallest stored band, use the greatest lower band.
        selected = max(lower) if lower else min(self.resolutions)
        return self.resolution_index[selected]

    def _char_index(self, char: str) -> int:
        if len(char) != 1:
            raise ValueError("char must contain exactly one Unicode character")
        try:
            return self.character_index[char]
        except KeyError as error:
            raise KeyError(f"character is not in this index: {char!r}") from error

    def _family_index(self, family: str) -> int:
        try:
            return self.family_index[family]
        except KeyError as error:
            raise KeyError(f"unknown family key: {family}") from error

    def _face_index(self, face: str) -> int:
        try:
            return self.face_index[face]
        except KeyError as error:
            raise KeyError(f"unknown face id: {face}") from error

    def face_distance(self, face_a: str, face_b: str, char: str, resolution: int) -> dict[str, object]:
        face_a_index = self._face_index(face_a)
        face_b_index = self._face_index(face_b)
        resolution_index = self._resolution_index(resolution)
        char_index = self._char_index(char)
        covered = bool(self.coverage[face_a_index, char_index] and self.coverage[face_b_index, char_index])
        if resolution_index is None:
            return self._result(face_a, face_b, char, None, None, covered, STATUS_UNKNOWN_RESOLUTION, int(resolution))
        actual_resolution = self.resolutions[resolution_index]
        if not covered:
            distance = None
            status = STATUS_MISSING
        else:
            distance_value = float(self.face_distance_table[face_a_index, face_b_index, resolution_index, char_index])
            distance = distance_value
            status = STATUS_EXACT_BITMAP if distance_value <= 0.0 else STATUS_LOW_DISTANCE if distance_value <= float(self.metadata["low_distance_threshold"]) else STATUS_DIFFERENT
        return self._result(face_a, face_b, char, actual_resolution, distance, covered, status, int(resolution))

    def pair_distance(self, family_a: str, family_b: str, char: str, resolution: int) -> dict[str, object]:
        family_a_index = self._family_index(family_a)
        family_b_index = self._family_index(family_b)
        resolution_index = self._resolution_index(resolution)
        char_index = self._char_index(char)
        if resolution_index is None:
            members_a = self.families[family_a_index]["member_indices"]
            members_b = self.families[family_b_index]["member_indices"]
            covered = bool(self.coverage[np.asarray(members_a), char_index].any() and self.coverage[np.asarray(members_b), char_index].any())
            return self._result(family_a, family_b, char, None, None, covered, STATUS_UNKNOWN_RESOLUTION, int(resolution))
        actual_resolution = self.resolutions[resolution_index]
        status = int(self.family_status_table[family_a_index, family_b_index, resolution_index, char_index])
        distance_value = self.family_distance_table[family_a_index, family_b_index, resolution_index, char_index]
        covered = status != STATUS_MISSING
        distance = float(distance_value) if covered and np.isfinite(distance_value) else None
        return self._result(family_a, family_b, char, actual_resolution, distance, covered, status, int(resolution))

    def _result(self, a: str, b: str, char: str, resolution: int | None, distance: float | None, covered: bool, status: int, requested_resolution: int) -> dict[str, object]:
        return {
            "a": a,
            "b": b,
            "char": char,
            "resolution": None if resolution is None else int(resolution),
            "requested_resolution": requested_resolution,
            "distance": distance,
            "covered": covered,
            "shared": None if not covered or status == STATUS_UNKNOWN_RESOLUTION else status in (STATUS_EXACT_BITMAP, STATUS_LOW_DISTANCE, STATUS_SAME_FAMILY),
            "status": STATUS_NAMES[status],
            "low_resolution_approximation": status == STATUS_LOW_DISTANCE and resolution is not None and int(resolution) <= 24,
            "evidence_basis": "rendered_raster",
            "vector_glyph_equal": None,
        }

    def image(self, face: str, char: str, resolution: int | None = None) -> np.ndarray | None:
        face_index = self._face_index(face)
        char_index = self._char_index(char)
        resolution = self.source_resolution if resolution is None else int(resolution)
        key = f"glyph_r{resolution}"
        if key not in self._arrays:
            raise KeyError(f"resolution is not stored: {resolution}")
        if not bool(self.coverage[face_index, char_index]):
            return None
        if key not in self._glyph_cache:
            self._glyph_cache[key] = self._arrays[key]
        return np.asarray(self._glyph_cache[key][face_index, char_index]).copy()


def load_index(path: str | Path = ROOT / "data/glyph-discrimination-v1") -> GlyphDiscriminationIndex:
    """Load an index without importing Pillow; ``path`` may be its directory or JSON."""

    candidate = Path(path)
    index_path = candidate if candidate.suffix == ".json" else candidate / "index.json"
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    return GlyphDiscriminationIndex(index_path, metadata)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/glyph-discrimination-v1")
    parser.add_argument("--extra-chars", action="append", default=[], metavar="FILE_OR_TEXT", help="UTF-8 character file, or literal characters; repeatable")
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--source-resolution", type=int, default=DEFAULT_SOURCE_RESOLUTION)
    parser.add_argument("--query", nargs=4, metavar=("FAMILY_A", "FAMILY_B", "CHAR", "RESOLUTION"), help="query an existing index instead of building")
    args = parser.parse_args()
    if args.query:
        index = load_index(args.output)
        family_a, family_b, char, resolution = args.query
        print(json.dumps(index.pair_distance(family_a, family_b, char, int(resolution)), ensure_ascii=False, indent=2))
        return
    index = build_index(args.output, args.extra_chars, args.resolutions, args.source_resolution)
    print(json.dumps(index["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
