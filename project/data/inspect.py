import json, collections, glob

for path in ["bh-annotation.json", "sm-annotation.json", "bh-phone-annotation.json"]:
    d = json.load(open(path))
    meta = d["_via_img_metadata"].values()
    labeled = [v for v in meta if v["regions"]]
    cnt = collections.Counter(r["region_attributes"].get("hold_type")
                              for v in labeled for r in v["regions"])
    shapes = collections.Counter(r["shape_attributes"]["name"]
                                 for v in labeled for r in v["regions"])
    print(path, "| images:", len(d["_via_img_metadata"]),
          "| with regions:", len(labeled), "|", dict(cnt), dict(shapes))





for path in ["bh-annotation.json", "sm-annotation.json", "bh-phone-annotation.json"]:
    d = json.load(open(path))
    print("==", path)
    for v in d["_via_img_metadata"].values():
        if v["regions"]:
            print(v["filename"], len(v["regions"]))
