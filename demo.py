"""Run each stage of the RAG pipeline from the command line. See README.md for the walkthrough.

    python demo.py nosources "question"    # stage 1: LLM with no sources
    python demo.py chunks                  # stage 2: how the policies were split
    python demo.py search "question"       # stages 3-5: compare retrieval methods
    python demo.py ask "question"          # stages 6-7: the full pipeline
    python demo.py eval                    # stage 8: measure it
"""

import argparse
import json
import random

import rag


def cmd_nosources(args):
    print(rag.ask_without_sources(args.question))


def cmd_chunks(args):
    chunks = rag.build_chunks()
    sizes = sorted(len(c.body.split()) for c in chunks)
    print(f"{len(chunks)} chunks from 16 policies; words per chunk: "
          f"min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}\n")
    for c in random.Random(args.seed).sample(chunks, args.show):
        print("-" * 80)
        print(c.text[:800])
    print("-" * 80)


def print_hits(hits):
    for rank, (chunk, score) in enumerate(hits, 1):
        print(f"  {rank}. {score:7.3f}  {chunk.doc['title'][:55]}  >  {chunk.section[-45:]}")


def cmd_search(args):
    index = rag.Index()
    for mode in [args.mode] if args.mode else rag.MODES:
        print(f"\n[{mode}]")
        print_hits(index.search(args.question, mode))


def cmd_ask(args):
    result = rag.ask(rag.Index(), args.question)
    print("Retrieved (reranker score):")
    print_hits(result["hits"])
    print(f"\nOutcome: {result['outcome']}\n")
    print(result["answer"])
    if result["cited"]:
        print("\nSources:")
        for chunk in result["cited"]:
            print(f"  - {chunk.doc['title']} > {chunk.section}\n    {chunk.doc['source_url']}")


def cmd_eval(args):
    """Score each retrieval method and the gate on data/eval.json. No LLM calls, so it's free."""
    index = rag.Index()
    questions = json.load(open("data/eval.json"))
    answerable = [q for q in questions if q["expected"]]
    results = [{mode: index.search(q["question"], mode) for mode in rag.MODES} for q in questions]

    def is_right(chunk, q):
        return q["expected"] in chunk.doc["title"]

    print(f"Retrieval, {len(answerable)} answerable questions (is the expected policy found?)")
    print(f"  {'method':8}  {'top-1':>6}  {'top-5':>6}")
    for mode in rag.MODES:
        pairs = [(q, r[mode]) for q, r in zip(questions, results) if q["expected"]]
        top1 = sum(is_right(hits[0][0], q) for q, hits in pairs)
        top5 = sum(any(is_right(c, q) for c, _ in hits) for q, hits in pairs)
        print(f"  {mode:8}  {top1:>3}/{len(pairs)}  {top5:>3}/{len(pairs)}")

    print(f"\nGate at {args.gate} (best reranker score below it = refer, don't answer)")
    answered = referred = 0
    mistakes = []
    for q, r in zip(questions, results):
        score = r["rerank"][0][1]
        if q["expected"] and score >= args.gate:
            answered += 1
        elif not q["expected"] and score < args.gate:
            referred += 1
        else:
            mistakes.append(f"  {score:.4f}  {'referred' if q['expected'] else 'answered'}: {q['question']}")
    print(f"  answerable questions answered:    {answered}/{len(answerable)}")
    print(f"  unanswerable questions referred:  {referred}/{len(questions) - len(answerable)}")
    if mistakes:
        print("Gate mistakes:\n" + "\n".join(mistakes))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("nosources", help="stage 1: ask the LLM with no sources")
    p.add_argument("question")
    p.set_defaults(func=cmd_nosources)

    p = sub.add_parser("chunks", help="stage 2: show chunk stats and a few sample chunks")
    p.add_argument("--show", type=int, default=3, help="how many sample chunks to print")
    p.add_argument("--seed", type=int, default=1, help="change it to see different samples")
    p.set_defaults(func=cmd_chunks)

    p = sub.add_parser("search", help="stages 3-5: compare retrieval methods for a question")
    p.add_argument("question")
    p.add_argument("--mode", choices=rag.MODES, help="show only one method")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="stages 6-7: the full pipeline, answer with citations")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="stage 8: score retrieval and the gate on data/eval.json")
    p.add_argument("--gate", type=float, default=rag.GATE_THRESHOLD, help="try a different gate threshold")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
