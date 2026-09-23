"""Font/style metadata independent from the legacy proxy-answer policy.

The v1 ``family_policy`` remains the compatibility contract for frozen
experiments.  This module describes source-family hypotheses, coverage and
Chinese replacement compatibility for the opt-in style-v2 path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyDefinition:
    key: str
    name: str
    preferred_replacement: str | None
    compatible_replacements: tuple[str, ...] = ()
    replacement_groups: tuple[str, ...] = ()
    vendor_coverage: str = "unknown"
    kana_style: tuple[str, ...] = ()
    kanji_style: tuple[str, ...] = ()


DEFINITIONS = (
    ("PShinMGoPr6N-", FamilyDefinition("shinmaru", "新丸ゴ", "圆体",
                                      kana_style=("rounded",), kanji_style=("rounded-gothic",))),
    ("PRyuminPr6N-", FamilyDefinition("ryumin", "Ryumin", "宋体",
                                     kana_style=("mincho",), kanji_style=("stroke-contrast",))),
    ("PAntiqueANpProN-", FamilyDefinition("antique", "Antique AN+", "黑体")),
    # Leave script style unset: the family name does not establish whether
    # its Japanese kana are Gothic-, Mincho-, or otherwise shaped.
    ("YWHeiTi-", FamilyDefinition("ywheiti", "攸望黑体（日文）", "黑体")),
    ("SkipStd-", FamilyDefinition("folk_skip", "Skip / Folk", "润圆/雅士黑", (),
                                 ("folk-skip-family",), kana_style=("informal",))),
    ("ManyoKoinLargeStd-", FamilyDefinition("koin", "Manyo Koin Large", "淡古印")),
    ("ComicMysteryStd-", FamilyDefinition("mystery", "Comic Mystery", "神秘体")),
    ("ComicReggaeStd-", FamilyDefinition("reggae", "Comic Reggae", "雷盖/雷鬼体")),
    ("FolkPro-", FamilyDefinition("folk_skip", "Skip / Folk", "润圆/雅士黑", (),
                                 ("folk-skip-family",), kana_style=("informal",))),
    ("TakeStd-", FamilyDefinition("take", "Take", "竹体")),
    ("HummingStd-", FamilyDefinition("humming", "Humming", "轻吟体",
                                    replacement_groups=("soft-rounded",),
                                    vendor_coverage="partial", kana_style=("soft", "rounded"),
                                    kanji_style=("soft-gothic",))),
    ("SeuratProN-", FamilyDefinition("seurat", "Seurat", "圆体",
                                    kana_style=("rounded",), kanji_style=("rounded-gothic",))),
    ("YWJiaoMinTi-", FamilyDefinition("yw_jiaoming", "攸望角明体", "角明体",
                                     kana_style=("mincho",), kanji_style=("stroke-contrast",))),
    ("YWJiangYuanTi-", FamilyDefinition("yw_jiangyuan", "攸望降圆体", "降圆体",
                                       kana_style=("rounded",), kanji_style=("rounded-gothic",))),
)


def _weight_class(value):
    if value is None:
        return "unknown"
    value = int(value)
    if value <= 350:
        return "light"
    if value <= 550:
        return "regular"
    if value <= 750:
        return "bold"
    return "ultrabold"


def _rodin_weight(postscript, metadata):
    suffix = postscript.rsplit("-", 1)[-1]
    known = {"L": (300, "light"), "B": (700, "bold"), "UB": (800, "ultrabold")}
    if suffix in known:
        return known[suffix]
    value = metadata.get("weight")
    return value, _weight_class(value)


def _named_weight_class(postscript, metadata):
    """Normalize coarse weight from explicit face names before OS/2 fallback."""
    style = str(metadata.get("style") or metadata.get("typographic_style") or "").upper()
    suffix = postscript.rsplit("-", 1)[-1].upper()
    names = (style, suffix)
    if any(name in {"ULTRA", "ULTRABOLD", "UB", "U", "BLACK"} for name in names):
        return "ultrabold", "sfnt_style_name"
    if any(name in {"BOLD", "B", "DB", "DEMI", "DEMIBOLD"} or name.startswith("B-") for name in names):
        return "bold", "sfnt_style_name"
    if any(name in {"LIGHT", "L", "EXTRALIGHT", "THIN"} or name.startswith("L-") for name in names):
        return "light", "sfnt_style_name"
    if any(name in {"MEDIUM", "M", "REGULAR", "R"} or name.startswith("M-") for name in names):
        return "regular", "sfnt_style_name"
    return _weight_class(metadata.get("weight")), "os2_weight_class"


def font_style_policy(postscript, folder="", metadata=None):
    """Return v2 source-family, coverage and replacement metadata."""
    metadata = metadata or {}
    if postscript.startswith("RodinHappyPro-"):
        weight_value, weight_class = _rodin_weight(postscript, metadata)
        weight_source = "postscript_suffix"
        definition = FamilyDefinition(
            "rodin_happy", "RodinHappy", None,
            replacement_groups=("rodin-rounded",),
            kana_style=("rodin-happy", "rounded"),
            kanji_style=("rounded-gothic",),
        )
    else:
        definition = next((value for prefix, value in DEFINITIONS if postscript.startswith(prefix)), None)
        if definition is None:
            raise ValueError(f"Unmapped font: {postscript} in {folder}")
        weight_value = metadata.get("weight")
        weight_class, weight_source = _named_weight_class(postscript, metadata)
    preferred = definition.preferred_replacement
    replacements = ([{"name": preferred, "role": "preferred"}] if preferred else [])
    replacements.extend({"name": name, "role": "compatible"}
                        for name in definition.compatible_replacements if name != preferred)
    warnings = []
    if definition.vendor_coverage == "partial":
        warnings.append("vendor_variant_uncovered")
    return {
        "face_id": metadata.get("id") or metadata.get("sha256", "")[:16] or postscript,
        "postscript": postscript,
        "source_family": {"key": definition.key, "name": definition.name},
        "vendor_variant": None,
        "weight_value": weight_value,
        "weight_class": weight_class,
        "weight_source": weight_source,
        "script_style": {"kana": list(definition.kana_style), "kanji": list(definition.kanji_style)},
        "coverage": {"vendor": definition.vendor_coverage,
                     "weights": [weight_class] if weight_class != "unknown" else []},
        "replacement_groups": list(definition.replacement_groups),
        "replacement_candidates": replacements,
        "coverage_warnings": warnings,
    }


def replacement_candidates(policy, observed_style=None):
    """Return replacement candidates, adding Rodin choices from observed style."""
    rows = [dict(row) for row in policy.get("replacement_candidates", [])]
    if policy["source_family"]["key"] != "rodin_happy":
        return rows
    style = observed_style or {}
    weight = style.get("weight_class") or policy.get("weight_class")
    rounded = style.get("kanji_roundness")
    preferred = "少女"
    if weight in ("bold", "semibold") or rounded == "moderate":
        preferred = "方圆"
    if weight in ("ultrabold", "black"):
        preferred = "海报体"
    order = [preferred] + [name for name in ("少女", "方圆", "海报体") if name != preferred]
    return [{"name": name, "role": "preferred" if i == 0 else "compatible"}
            for i, name in enumerate(order)]
