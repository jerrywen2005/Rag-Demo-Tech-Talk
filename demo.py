"""Run each stage of the RAG pipeline from the command line. See README.md for the walkthrough.

    python demo.py nosources "question"    # stage 1: LLM with no sources
    python demo.py chunks                  # stage 2: how the policies were split
    python demo.py search "question"       # stages 3-5: compare retrieval methods
"""

import argparse
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
