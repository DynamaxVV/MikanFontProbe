"""Explicit font-family identity, independent of displayed Chinese answer."""


def family_policy(postscript, folder):
    if postscript.startswith("RodinHappyPro-"):
        variants = {"L": "少女", "B": "方圆", "UB": "海报体"}
        weight = postscript.rsplit("-", 1)[1]
        if weight not in variants:
            raise ValueError(f"Unmapped RodinHappy weight: {weight}")
        return {"key": postscript, "name": f"RodinHappy {weight}", "answer": variants[weight]}
    definitions = [
        ("PShinMGoPr6N-", "shinmaru", "新丸ゴ", "圆体"),
        ("PRyuminPr6N-", "ryumin", "Ryumin", "宋体"),
        ("PAntiqueANpProN-", "antique", "Antique AN+", "黑体"),
        ("YWHeiTi-", "ywheiti", "攸望黑体（日文）", "黑体"),
        ("SkipStd-", "skip", "Skip", "润圆/雅士黑"),
        ("ManyoKoinLargeStd-", "koin", "Manyo Koin Large", "淡古印"),
        ("ComicMysteryStd-", "mystery", "Comic Mystery", "神秘体"),
        ("ComicReggaeStd-", "reggae", "Comic Reggae", "雷盖/雷鬼体"),
        ("FolkPro-", "folk", "Folk", "方新书"),
        ("TakeStd-", "take", "Take", "竹体"),
        ("HummingStd-", "humming", "Humming", "轻吟体"),
    ]
    for prefix,key,name,answer in definitions:
        if postscript.startswith(prefix):
            return {"key": key, "name": name, "answer": answer}
    raise ValueError(f"Unmapped font: {postscript} in {folder}")
