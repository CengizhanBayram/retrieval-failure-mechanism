"""
D - haystack-source check for the stored retrieval-head sets
(merged-paper follow-up; EXPLORATORY).

Location: this script lives in retrieval-failure-mechanism so that every
follow-up runs from one repository, but it measures the Part I detector of
retrieval-head-profile. It runs against a clone of that repository, given by
``--rhp-repo`` or ``$RHP_PART2_REPO`` (the Colab setup cells set it to
/content/rope-part2); the detector code, configs/panel.yaml and the stored
profiles (datas/results) are all read from there, and nothing in it is written.

Why: the inherited corpus loader (Part-1 ``src/corpus.py``) requests PG-19 with
``load_dataset("pg19", trust_remote_code=True)``. Under the ``datasets`` version
the profile runs record (4.0.0) dataset scripts are no longer executed, so the
request most probably failed and the loader silently used its fallback pool of
50 neutral sentences repeated 40 times. Nothing in the stored profiles says which
pool was used, and the planned haystack-source control (R7) was never run. This
script settles both questions empirically, with the profile's own detectors and
settings (4096 tokens, 200 argmax probes + 100 copy-score probes, depths
{0.1, 0.25, 0.5, 0.75, 0.9}, seed 42, the panel's 8-/4-bit loading):

  --haystack fallback  re-detect with the fallback pool FORCED. If the stored
                       sets came from this pool, the probes are byte-identical
                       and the re-detection reproduces the stored score
                       matrices up to GPU noise (Jaccard ~ 1, Spearman ~ 1).
  --haystack wikitext  re-detect on a natural haystack (WikiText-103, 5,000
                       sentences, the same sentence filter as the PG-19 path).
                       J(stored, wikitext) / R_self says how much of the head
                       identity depends on the haystack.
  --haystack pg19      the corpus the code intended, from a Parquet mirror of
                       PG-19 (no loading script); aborts if no mirror loads.

The pool is INSTALLED into ``src.corpus._CORPUS_CACHE`` before the model is
loaded and verified (sha256) before and after detection, so the run can never
silently fall back to another pool. Every output records the pool's hash, size
and first sentences.

Outputs: ``<results>/robustness/D_haystack/<haystack>/<model>_seed<seed>.json``
and, with ``--summarize``, ``<results>/robustness/D_haystack/summary.json``.

Examples (Colab, one A100/L4 session; ~10 min per 7-9B model; RHP_PART2_REPO set):
    python scripts/rhp_haystack_check.py --models qwen25_7b llama31_8b --haystack fallback
    python scripts/rhp_haystack_check.py --models core partII --haystack wikitext
    python scripts/rhp_haystack_check.py --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path



def rhp_root(cli_path: str | None = None) -> Path:
    """The retrieval-head-profile clone this check runs against; put it on sys.path
    so that ``rhp.*`` imports resolve there."""
    p = cli_path or os.environ.get("RHP_PART2_REPO")
    if not p:
        raise SystemExit("set --rhp-repo (or $RHP_PART2_REPO) to a retrieval-head-profile clone")
    root = Path(p).expanduser().resolve()
    if not (root / "rhp").is_dir() or not (root / "scripts" / "_common.py").is_file():
        raise SystemExit(f"{root} is not a retrieval-head-profile clone (rhp/ or scripts/_common.py missing)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def rhp_common(root: Path):
    """retrieval-head-profile's scripts/_common.py, loaded from its file so that it
    cannot be confused with this repository's own scripts/_common.py."""
    spec = importlib.util.spec_from_file_location("rhp_scripts_common", root / "scripts" / "_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hf_files():
    """This repository's src/failure_mech/hf_files.py, loaded from its file (it has
    no package imports), so the clone's sys.path setup is left untouched."""
    path = Path(__file__).resolve().parents[1] / "src" / "failure_mech" / "hf_files.py"
    spec = importlib.util.spec_from_file_location("rfm_hf_files", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("haystack_check")

CORE = ["llama32_3b", "llama31_8b", "qwen25_3b", "qwen25_7b", "gemma2_9b"]
PART2 = ["llama31_8b_instruct", "qwen25_7b_instruct", "gemma2_9b_it", "mistral7b_v03_instruct",
         "olmo2_7b_instruct", "phi35_mini", "qwen25_3b_instruct"]
GROUPS = {"core": CORE, "partII": PART2}
HAYSTACKS = ("fallback", "wikitext", "pg19")
# Parquet mirrors of PG-19 tried in order for --haystack pg19 (no loading script).
PG19_MIRRORS = [("emozilla/pg19", None), ("deepmind/pg19", None)]
TOPK = (5, 10, 20)


# ---------------------------------------------------------------------------
# Haystack pools
# ---------------------------------------------------------------------------

def _sentences_from_stream(ds, max_sentences: int, text_key: str = "text") -> list[str]:
    """The PG-19 branch of ``src.corpus.load_haystack_corpus``, verbatim: split on
    '. ', keep 20 < len < 200, re-append the period, stop at ``max_sentences``."""
    out: list[str] = []
    for example in ds:
        text = example.get(text_key) or ""
        for sent in text.split(". "):
            sent = sent.strip()
            if 20 < len(sent) < 200:
                out.append(sent + ".")
            if len(out) >= max_sentences:
                break
        if len(out) >= max_sentences:
            break
    return out


def build_pool(haystack: str, max_sentences: int = 5000) -> tuple[list[str], dict]:
    """Return (sentence pool, metadata). Raises if a natural pool cannot be built:
    a check that silently fell back would answer the wrong question."""
    import src.corpus as corpus

    if haystack == "fallback":
        pool = list(corpus.FALLBACK_SENTENCES) * 40          # exactly the loader's fallback
        meta = {"source": "src.corpus.FALLBACK_SENTENCES x 40"}
    elif haystack == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                          streaming=True)
        pool = _sentences_from_stream(ds, max_sentences)
        meta = {"source": "Salesforce/wikitext wikitext-103-raw-v1 (train, streamed)"}
    elif haystack == "pg19":
        from datasets import load_dataset
        pool, meta, errors = [], None, []
        for repo, cfg in PG19_MIRRORS:
            try:
                ds = load_dataset(repo, cfg, split="train", streaming=True) if cfg else \
                    load_dataset(repo, split="train", streaming=True)
                pool = _sentences_from_stream(ds, max_sentences)
                if len(pool) >= 500:
                    meta = {"source": f"{repo} (train, streamed)"}
                    break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{repo}: {exc}")
        if meta is None:
            raise RuntimeError("no PG-19 Parquet mirror could be loaded: " + " | ".join(errors))
    else:
        raise ValueError(f"unknown haystack {haystack!r}; expected one of {HAYSTACKS}")
    if haystack != "fallback" and len(pool) < 500:
        raise RuntimeError(f"{haystack}: only {len(pool)} sentences - refusing to run on a "
                           "degenerate pool.")
    meta.update(pool_meta(pool))
    return pool, meta


def pool_meta(pool: list[str]) -> dict:
    h = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()
    return {"n_sentences": len(pool), "n_unique": len(set(pool)), "sha256": h,
            "first_sentences": pool[:3]}


def install_pool(pool: list[str]) -> str:
    """Put ``pool`` into the process-wide cache both detectors read
    (``src.corpus.load_haystack_corpus`` returns the cache when it is set)."""
    import src.corpus as corpus
    corpus._CORPUS_CACHE = list(pool)
    got = corpus.load_haystack_corpus(max_sentences=len(pool))
    if got is not corpus._CORPUS_CACHE or pool_meta(got)["sha256"] != pool_meta(pool)["sha256"]:
        raise RuntimeError("haystack cache did not take the installed pool")
    return pool_meta(pool)["sha256"]


def assert_pool(expected_sha: str) -> None:
    import src.corpus as corpus
    cur = corpus._CORPUS_CACHE
    if cur is None or pool_meta(cur)["sha256"] != expected_sha:
        raise RuntimeError("haystack cache changed during the run - result discarded")


# ---------------------------------------------------------------------------
# Detection (the profile pipeline's E1 step, unchanged)
# ---------------------------------------------------------------------------

def detect_heads(model_key: str, model_cfg: dict, config: dict, *, seed: int,
                 context_length: int, n_samples: int | None, knockout: bool) -> dict:
    """Argmax + copy-score detection exactly as ``run_profile_for_model`` runs it
    (same classes, sample counts, positions, threshold, seed and loader)."""
    import gc

    import numpy as np
    import torch

    from src.model_loader import purge_hf_cache
    from src.repro import capture_environment, set_determinism
    from src.retrieval_head_detector import RetrievalHeadDetector
    from rhp.copy_score_detector import CopyScoreDetector
    from rhp.loader import load_model_any

    niah = config["niah"]
    threshold = niah.get("score_threshold", 0.1)
    n_samples = n_samples or niah.get("n_samples", 200)
    positions = niah["needle_positions"]
    set_determinism(seed, strict=config.get("reproducibility", {}).get("strict_determinism", False))
    t0 = time.time()
    model = tok = None
    try:
        # Mistral-7B-v0.3: the CDN refuses tokenizer.model at the pinned commit;
        # seed it from the byte-identical tokenizer.model.v3 (the loader is untouched).
        hf_files().ensure_tokenizer_files(model_cfg["hf_id"], model_cfg.get("revision") or "main")
        model, tok = load_model_any(model_cfg, model_key)
        det = RetrievalHeadDetector(model, tok, config, score_threshold=threshold, seed=seed)
        a_samples = det.generate_niah_samples(n_samples, [context_length], positions)
        argmax_scores = det.score_heads(a_samples)
        argmax_heads = det.get_retrieval_heads(argmax_scores)
        cdet = CopyScoreDetector(model, tok, config, score_threshold=threshold, seed=seed)
        c_samples = cdet.generate_samples(max(80, n_samples // 2), [context_length], positions)
        copy_scores = cdet.score_heads(c_samples)
        copy_heads = cdet.get_retrieval_heads(copy_scores)
        out = {
            "argmax_heads": [list(map(int, h)) for h in argmax_heads],
            "copy_heads": [list(map(int, h)) for h in copy_heads],
            "argmax_scores": np.asarray(argmax_scores).tolist(),
            "copy_scores": np.asarray(copy_scores).tolist(),
            "n_layers": int(model.config.num_hidden_layers),
            "n_heads": int(model.config.num_attention_heads),
            "probe_digest": hashlib.sha256(
                "\n".join(s.get("prompt", s.get("text", "")) if isinstance(s, dict) else str(s)
                          for s in a_samples[:20]).encode("utf-8")).hexdigest(),
        }
        if knockout and argmax_heads:
            from rhp.knockout import KnockoutEvaluator
            ko = KnockoutEvaluator(model, tok, config)
            ko_samples = det.generate_niah_samples(80, [context_length], [0.25, 0.5, 0.75])
            out["knockout"] = ko.run(ko_samples, argmax_heads, out["n_heads"], out["n_layers"],
                                     random_seeds=config["profile"].get("knockout_random_seeds",
                                                                         [0, 1, 2, 3, 4]))
        out["elapsed_sec"] = round(time.time() - t0, 1)
        out["environment"] = capture_environment()
        return out
    finally:
        model = tok = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            purge_hf_cache(model_cfg["hf_id"])
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Comparison with the stored profile
# ---------------------------------------------------------------------------

def _heads(lst) -> set:
    return {(int(l), int(h)) for (l, h) in lst}


def _ranked(heads: set, scores) -> list:
    return sorted(heads, key=lambda lh: (-float(scores[lh[0]][lh[1]]), lh[0], lh[1]))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def compare(new: dict, stored: dict) -> dict:
    """Head-set and score agreement between a re-detection and a stored profile."""
    import numpy as np
    from scipy import stats

    out = {}
    for det in ("argmax", "copy"):
        A, B = _heads(stored[f"{det}_heads"]), _heads(new[f"{det}_heads"])
        Sa = np.asarray(stored[f"{det}_scores"], dtype=float)
        Sb = np.asarray(new[f"{det}_scores"], dtype=float)
        rho = float(stats.spearmanr(Sa.ravel(), Sb.ravel())[0]) if Sa.shape == Sb.shape else float("nan")
        ra, rb = _ranked(A, Sa), _ranked(B, Sb)
        out[det] = {
            "jaccard": jaccard(A, B), "n_stored": len(A), "n_new": len(B),
            "spearman_scores": rho,
            "max_abs_score_diff": float(np.max(np.abs(Sa - Sb))) if Sa.shape == Sb.shape else float("nan"),
            "top_k_overlap": {str(k): len(set(ra[:k]) & set(rb[:k])) for k in TOPK
                              if k <= min(len(ra), len(rb))},
        }
    return out


def stored_profile(results_root: Path, model_key: str, seed: int) -> dict | None:
    p = results_root / "profile" / f"{model_key}_seed{seed}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rself_table(results_root: Path) -> dict:
    """Test-retest Jaccard (seed 42 vs 123) of the CORE models at the stored
    threshold, plus the panel medians used as the fallback reference - the same
    definition as the merged paper's decision rules."""
    import numpy as np
    rs = {"argmax": {}, "copy": {}}
    for m in CORE:
        a, b = stored_profile(results_root, m, 42), stored_profile(results_root, m, 123)
        if a and b:
            for det in rs:
                rs[det][m] = jaccard(_heads(a[f"{det}_heads"]), _heads(b[f"{det}_heads"]))
    med = {det: (float(np.median(list(v.values()))) if v else float("nan")) for det, v in rs.items()}
    return {"per_model": rs, "median": med}


def summarize(results_root: Path, out_root: Path, seed: int) -> dict:
    """Collect every per-model output under ``out_root/D_haystack`` into
    ``summary.json``; R_self comes from the stored profiles under
    ``results_root/profile``."""
    base = out_root / "D_haystack"
    rs = rself_table(results_root)
    table = {}
    for hs in HAYSTACKS:
        d = base / hs
        if not d.is_dir():
            continue
        for f in sorted(d.glob(f"*_seed{seed}.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            m = r["model"]
            row = table.setdefault(m, {})
            cmp_ = r.get("comparison") or {}
            entry = {"pool_sha256": r["haystack"]["sha256"], "n_sentences": r["haystack"]["n_sentences"]}
            for det in ("argmax", "copy"):
                c = cmp_.get(det)
                if not c:
                    continue
                R = rs["per_model"][det].get(m, rs["median"][det])
                entry[det] = {**c, "R_self": R,
                              "J_over_Rself": c["jaccard"] / R if R and R == R else float("nan")}
            if "knockout" in r:
                entry["knockout"] = r["knockout"]
            row[hs] = entry
    out = {"rself": rs, "models": table,
           "note": "fallback rows test whether the stored sets came from the fallback pool "
                   "(expect J ~ 1, Spearman ~ 1 if they did); wikitext/pg19 rows measure "
                   "haystack dependence relative to R_self."}
    (base).mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="D: haystack-source check of the stored head sets.")
    ap.add_argument("--models", nargs="*", default=[],
                    help="model keys or groups: core, partII (the seven Part II models)")
    ap.add_argument("--haystack", choices=HAYSTACKS, default="wikitext")
    ap.add_argument("--n-sentences", type=int, default=5000,
                    help="natural-pool size (the PG-19 path of the loader uses 5000)")
    ap.add_argument("--rhp-repo", default=None,
                    help="retrieval-head-profile clone (default: $RHP_PART2_REPO)")
    ap.add_argument("--config", default=None,
                    help="panel config (default: <rhp-repo>/configs/panel.yaml)")
    ap.add_argument("--part1-repo", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="root holding profile/ (stored artifacts) and robustness/ "
                         "(default: <rhp-repo>/datas/results)")
    ap.add_argument("--out-dir", default=None,
                    help="where D_haystack/ is written (default: <results-dir>/robustness)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--knockout", action="store_true",
                    help="also run the knockout on the re-detected argmax set (slower)")
    ap.add_argument("--time-budget-hours", type=float, default=23.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="only (re)build summary.json from the per-model outputs")
    return ap.parse_args(argv)


def expand_models(spec: list[str]) -> list[str]:
    out: list[str] = []
    for s in spec:
        for m in GROUPS.get(s, [s]):
            if m not in out:
                out.append(m)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    project = rhp_root(args.rhp_repo)
    results_root = Path(args.results_dir) if args.results_dir else project / "datas" / "results"
    out_root = Path(args.out_dir) if args.out_dir else results_root / "robustness"
    if args.summarize:
        s = summarize(results_root, out_root, args.seed)
        logger.info("summary written (%d models).", len(s["models"]))
        return 0

    common = rhp_common(project)
    bootstrap, save_json = common.bootstrap, common.save_json
    bootstrap(args.part1_repo)
    from rhp.panel import load_panel, model_cfg

    config = load_panel(args.config or str(project / "configs" / "panel.yaml"))
    models = expand_models(args.models)
    if not models:
        raise SystemExit("no models given (use keys or the groups core / partII)")
    pool, meta = build_pool(args.haystack, args.n_sentences)
    sha = install_pool(pool)
    logger.info("haystack %s installed: %d sentences (%d unique), sha256 %s…",
                args.haystack, meta["n_sentences"], meta["n_unique"], sha[:12])
    out_dir = out_root / "D_haystack" / args.haystack
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    failed: dict[str, str] = {}
    for key in models:
        out = out_dir / f"{key}_seed{args.seed}.json"
        if out.exists() and not args.overwrite:
            logger.info("[%s] exists -> skip.", key)
            continue
        if time.time() - start > args.time_budget_hours * 3600:
            logger.warning("time budget reached before %s; re-run to resume.", key)
            break
        assert_pool(sha)
        try:
            res = detect_heads(key, model_cfg(config, key), config, seed=args.seed,
                               context_length=args.context, n_samples=args.n_samples,
                               knockout=args.knockout)
        except Exception as exc:  # noqa: BLE001  one model must not cost the others
            logger.exception("[%s] FAILED, continuing with the next model.", key)
            failed[key] = f"{type(exc).__name__}: {exc}"
            continue
        assert_pool(sha)
        stored = stored_profile(results_root, key, args.seed)
        rec = {"model": key, "seed": args.seed, "context_length": args.context,
               "status": "exploratory follow-up D", "haystack": {"name": args.haystack, **meta},
               **res,
               "comparison": compare(res, stored) if stored else None,
               "stored_profile": (f"profile/{key}_seed{args.seed}.json" if stored else None)}
        save_json(rec, out)
        if rec["comparison"]:
            c = rec["comparison"]
            logger.info("[%s] %s: J_argmax %.3f (rho %.3f) | J_copy %.3f (rho %.3f)", key,
                        args.haystack, c["argmax"]["jaccard"], c["argmax"]["spearman_scores"],
                        c["copy"]["jaccard"], c["copy"]["spearman_scores"])
    summarize(results_root, out_root, args.seed)
    if failed:
        for key, err in failed.items():
            logger.error("[%s] not done: %s", key, err[:300])
        logger.error("%d model(s) failed (summary written for the rest). Re-run the same "
                     "command to retry them; finished models are skipped.", len(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
