#!/usr/bin/env python3
"""A4 step 1: fetch every historical revision of gen9 randbats sets.json.

sets.json is require()d as a plain data file, so swapping it at runtime is enough to
realign the team generator with the version a replay was played on. No rebuilds needed.
"""
import json, os, time, urllib.request, hashlib

UA = "psrl-research/0.1 (non-commercial Pokemon RL research)"
REPO = "smogon/pokemon-showdown"
PATH = "data/random-battles/gen9/sets.json"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "sets_versions")


def get(url, tries=4):
    for a in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(r, timeout=30).read()
        except Exception as e:
            if a == tries - 1: raise
            print(f"  retry ({e}) in {2**a}s"); time.sleep(2 ** a)


def main():
    os.makedirs(OUT, exist_ok=True)
    commits, page = [], 1
    while True:
        u = (f"https://api.github.com/repos/{REPO}/commits?path={PATH}"
             f"&per_page=100&page={page}&since=2022-11-01T00:00:00Z")
        batch = json.loads(get(u))
        if not isinstance(batch, list):
            print("GitHub API said:", str(batch)[:200]); break
        if not batch: break
        commits += batch
        if len(batch) < 100: break
        page += 1; time.sleep(1)
    commits.sort(key=lambda c: c["commit"]["author"]["date"])
    print(f"{len(commits)} commits touching {PATH} since 2022-11")

    manifest = []
    for c in commits:
        sha, date = c["sha"], c["commit"]["author"]["date"]
        dest = os.path.join(OUT, f"{date[:10]}_{sha[:8]}.json")
        if not os.path.exists(dest):
            body = get(f"https://raw.githubusercontent.com/{REPO}/{sha}/{PATH}")
            with open(dest, "wb") as f:
                f.write(body)
            time.sleep(0.3)
        else:
            with open(dest, "rb") as f:
                body = f.read()
        manifest.append({
            "sha": sha, "date": date,
            "epoch": int(time.mktime(time.strptime(date, "%Y-%m-%dT%H:%M:%SZ"))),
            "file": os.path.basename(dest),
            "species": len(json.loads(body)),
            "md5": hashlib.md5(body).hexdigest()[:8],
            "msg": c["commit"]["message"].split("\n")[0][:60],
        })
        print(f"  {date[:10]} {sha[:8]} species={manifest[-1]['species']} {manifest[-1]['msg']}")

    # collapse consecutive identical contents
    dedup = [m for i, m in enumerate(manifest) if i == 0 or m["md5"] != manifest[i - 1]["md5"]]
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(os.path.getsize(os.path.join(OUT, m["file"])) for m in manifest)
    print(f"\n{len(manifest)} revisions ({len(dedup)} with distinct content), {total/1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
