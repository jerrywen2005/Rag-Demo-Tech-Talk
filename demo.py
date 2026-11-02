"""Run each stage of the RAG pipeline from the command line. See README.md for the walkthrough.

    python demo.py nosources "question"    # stage 1: LLM with no sources
    python demo.py chunks                  # stage 2: how the policies were split
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
