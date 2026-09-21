"""Evaluate the conservative guard on already-exposed regression evidence."""
import hashlib
import json

from mikan_font_probe.process_regions import ROOT
from mikan_font_probe.paths import resolve_repo_path
from mikan_font_probe.resolve_font_ambiguity import resolve

OUT = ROOT/"data/humming-v4"


def main():
    path = OUT/"guarded-results.json"
    if path.exists():
        raise ValueError("Preserve guard evaluation")
    old = json.loads((ROOT/"data/category-expansion-v3/results.json").read_text())["samples"]
    new = {r["qid"]: r for r in json.loads((OUT/"regression/results.json").read_text())["samples"]}
    rows = []
    for sample in old:
        policy = sample["variants"]["quality_7"]
        updated = new[sample["qid"]]["variants"]["quality_7"]
        assert policy["selected"] == updated["selected"]
        distinct = len({sample["glyphs"][i]["character"] for i in policy["selected"]})
        result = resolve(policy["ranking"], updated["ranking"], distinct)
        rows.append({"qid":sample["qid"],"expected":sample["expected"],"eligible":sample["eligible_seven"],
                     "baseline":policy["ranking"][0]["answer"],"result":result})
    controls = []
    for sample in json.loads((OUT/"folk-controls/results.json").read_text())["samples"]:
        result = resolve(sample["ranking"]["baseline"], sample["ranking"]["geometry"], len(sample["selected"]))
        controls.append({"qid":sample["qid"],"expected":sample["expected"],"eligible":sample["eligible_seven"],
                         "baseline":sample["ranking"]["baseline"][0]["answer"],"result":result})
    document = {"status":"guard designed after negative control failure; all results development evidence",
                "samples":rows,"controls":controls,"guard_sha256":hashlib.sha256((resolve_repo_path("resolve_font_ambiguity.py")).read_bytes()).hexdigest()}
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n")
    for group in (rows,controls):
        eligible = [r for r in group if r["eligible"]]
        print({"n":len(eligible),"old":sum(r['baseline']==r['expected'] for r in eligible),
               "new":sum(r['result']['ranking'][0]['answer']==r['expected'] for r in eligible)})
        print([(r["qid"],r["baseline"],r["result"]["ranking"][0]["answer"],r["result"]["state"]) for r in group if r["result"]["changed"]])


if __name__ == "__main__":
    main()
