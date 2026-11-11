"""Rejoue en temps réel une partie enregistrée par match_recorder contre goal_service.py (analyse en continu).

    python replay_match_test.py <dossier_match> [premier_but] [nombre_de_buts] [port]

Envoie les fragments au rythme d'origine (comme le ferait le Pi), appelle /goal à l'heure de chaque but compté
(événement score_goal), et compare le type trouvé à celui de l'analyse par clip (goals_analysis.json s'il existe).
"""
import http.client
import json
import sys
import threading
import time
from pathlib import Path


def read_jsonl(p):
    out = []
    for l in open(p, encoding="utf-8-sig", errors="replace"):
        l = l.strip().lstrip("\x00")
        if l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def main(match_dir, first=0, count=3, port=8766):
    d = Path(match_dir)
    chunks, events = read_jsonl(d / "chunks.jsonl"), read_jsonl(d / "events.jsonl")
    sizes = [c["bytes"] for c in chunks]
    off = [0]
    for s in sizes:
        off.append(off[-1] + s)
    goals = [e for e in events if e["event"] == "score_goal"]
    sel = goals[first:first + count]
    t_from, t_to = sel[0]["ts"] - 12, sel[-1]["ts"] + 6
    idx = [i for i, c in enumerate(chunks) if t_from <= c["ts"] <= t_to and not c.get("lead")]
    i0, i1 = idx[0], idx[-1]
    print("fragments %d..%d (%.0f s), %d buts" % (i0, i1, chunks[i1]["ts"] - chunks[i0]["ts"], len(sel)), flush=True)

    ref = {}
    try:
        ref = json.load(open(d.parent.parent / "goals_analysis.json"))
    except Exception:
        pass
    journal = read_jsonl(d.parent.parent / "goals" / "journal.jsonl")
    gid = lambda ts: next((e["id"] for e in journal if e["event"] == "goal" and ts - 1 <= e["ts"] <= ts + 12), None)

    f = open(d / "match.h264", "rb")
    t0 = time.time() + 1.0
    base = chunks[i0]["mtime"]   # heure d'écriture par la caméra (régulière), pas celle de la copie par le Pi
    results = []

    def call_goal(ev, last_seq):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        t = time.time()
        c.request("POST", "/goal?last_seq=%d" % last_seq, body=b"")
        res = json.loads(c.getresponse().read())
        res["latency"] = round(time.time() - t, 2)
        res["ts"] = ev["ts"]
        results.append(res)

    timers = []
    for ev in sel:
        last_seq = max(i for i in range(i0, i1 + 1) if chunks[i]["mtime"] <= ev["ts"])
        delay = t0 + (ev["ts"] - base) - time.time()
        tm = threading.Timer(delay, call_goal, args=(ev, last_seq))
        tm.start()
        timers.append(tm)

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    late = []
    for i in range(i0, i1 + 1):
        wait = t0 + (chunks[i]["mtime"] - base) - time.time()
        if wait > 0:
            time.sleep(wait)
        f.seek(off[i])
        data = f.read(sizes[i])
        t = time.time()
        conn.request("POST", "/fragment?seq=%d" % i, body=data)
        conn.getresponse().read()
        late.append(time.time() - t)
    for tm in timers:
        tm.join()
    time.sleep(3)
    print("envoi d'un fragment: médiane %.0f ms, max %.0f ms" % (1000 * sorted(late)[len(late) // 2], 1000 * max(late)))
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/health")
    st = json.loads(c.getresponse().read())["stream"]
    print("service:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()},
          "-> détection %.0f ms par fragment (budget 200)" % (st["detect_ms"] / max(st["fragments"], 1)))
    for r in sorted(results, key=lambda r: r["ts"]):
        g = gid(r["ts"])
        b = ref.get("%d.h264" % g, {}).get("type") if g else None
        print("but %s: continu=%s (conf %.2f)  clip=%s  latence=%.2f s (attente %.2f, classement %.2f)  images=%s" % (
            time.strftime("%H:%M:%S", time.localtime(r["ts"])), r.get("type"), r.get("confidence") or 0, b, r["latency"],
            r.get("waited", 0), r.get("classify", 0), r.get("frames")))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 0, int(a[3]) if len(a) > 3 else 3, int(a[4]) if len(a) > 4 else 8766)
