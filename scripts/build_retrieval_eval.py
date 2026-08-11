"""Generate the paraphrase query set the retrieval eval runs on.

Why paraphrases: the lore index stores each record with its question stem, so
querying with the stem itself would hand the substring baseline the answer and
prove nothing. What actually happens in play is a player asking in their own
words. A strong model rewrites each stem into a colloquial player question
(different wording, same fact), and that becomes the gold query set: query in,
the record it was derived from is the one correct result.

The generated file is committed, so the eval is reproducible without a key and
the queries can be spot-checked by eye — both of which matter more than the
few cents regeneration costs.

Usage:
    python scripts/build_retrieval_eval.py             # all eras, zh
    python scripts/build_retrieval_eval.py --eras china
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chrono_agent.config import PROJECT_ROOT, Settings  # noqa: E402
from chrono_agent.factory import load_lore  # noqa: E402

OUT_PATH = PROJECT_ROOT / "eval" / "retrieval_queries.json"

PROMPT = """你在为一个检索评测集改写查询。下面是一条游戏内的历史知识记录：

原题面：{topic}
知识内容：{fact}

请把它改写成一个玩家在游戏里会随口问出的中文短问题，要求：
1. 问的是同一件事实，检索到这条记录应该能回答它；
2. 换一种问法——尽量不要照抄原题面里的措辞（专有名词实在绕不开可以保留）；
3. 口语化、简短，一句话；
4. 只输出这个问题本身，不要任何解释或引号。"""


async def _rewrite(
    client: httpx.AsyncClient, settings: Settings, topic: str, fact: str
) -> str:
    response = await client.post(
        f"{settings.deepseek_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
        json={
            "model": settings.deepseek_model,
            "messages": [
                {"role": "user", "content": PROMPT.format(topic=topic, fact=fact)}
            ],
            "temperature": 0.8,
            "max_tokens": 100,
            # Finding 02: thinking mode on by default -> empty replies.
            "thinking": {"type": "disabled"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip().strip('"「」')


async def build(eras: list[str], concurrency: int) -> dict:
    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        raise SystemExit("[!] DEEPSEEK_API_KEY missing — needed to generate queries.")

    semaphore = asyncio.Semaphore(concurrency)
    queries: dict[str, list] = {}

    async with httpx.AsyncClient() as client:

        async def one(era: str, index: int, entry: dict) -> None:
            topic = entry.get("_topic_zh", "")
            fact = entry.get("zh", "")
            if not topic or not fact:
                return
            async with semaphore:
                for attempt in (1, 2):
                    try:
                        query = await _rewrite(client, settings, topic, fact)
                        break
                    except httpx.HTTPError:
                        if attempt == 2:
                            print(f"    [!] {era}#{index}: gave up after retry")
                            return
                        await asyncio.sleep(2)
            if query:
                queries.setdefault(era, []).append({"doc": index, "query": query})

        for era in eras:
            entries = list(load_lore(era))
            print(f"{era}: rewriting {len(entries)} stems…")
            await asyncio.gather(
                *(one(era, i, entry) for i, entry in enumerate(entries))
            )
            queries[era] = sorted(queries.get(era, []), key=lambda q: q["doc"])
            print(f"{era}: {len(queries[era])} queries")

    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", default="all")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    from chrono_agent.config import DATA_DIR

    with (DATA_DIR / "lore.json").open(encoding="utf-8") as fh:
        available = sorted(json.load(fh))
    eras = (
        available
        if args.eras.strip().lower() == "all"
        else [e.strip() for e in args.eras.split(",") if e.strip()]
    )

    queries = asyncio.run(build(eras, args.concurrency))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(queries, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    total = sum(len(v) for v in queries.values())
    print(f"wrote {total} queries -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
