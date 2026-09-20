"""Check actual guarded inference against saved regression decisions."""
import json
from pathlib import Path
from rank_font_guarded import rank_sample

ROOT = Path(__file__).resolve().parent


def main():
    expected = json.loads((ROOT/"data/humming-v4/guarded-results.json").read_text())
    count = 0
    for directory, key in (("data/category-expansion-v3", "samples"),
                           ("data/humming-v4/folk-controls", "controls")):
        path = ROOT/directory
        samples = {s["qid"]: s for s in json.loads((path/"pools.json").read_text())["samples"]}
        for record in expected[key]:
            sample = samples[record["qid"]]
            result = rank_sample(sample, path/"templates")
            if sample.get("polarity") == "light":
                assert result["state"] == "unsupported_light_text_review_required"
            else:
                assert [r["key"] for r in result["top3"]] == [r["key"] for r in record["result"]["ranking"][:3]], record["qid"]
            assert result["auto_accept"] is False
            count += 1
    print(f"Verified {count} live entry-point decisions against preserved evidence; light text explicitly rejected")


if __name__ == "__main__":
    main()
