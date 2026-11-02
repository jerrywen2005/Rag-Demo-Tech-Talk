"""Run each stage of the RAG pipeline from the command line. See README.md for the walkthrough.

    python demo.py nosources "question"    # stage 1: LLM with no sources
"""

import argparse

import rag


def cmd_nosources(args):
    print(rag.ask_without_sources(args.question))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("nosources", help="stage 1: ask the LLM with no sources")
    p.add_argument("question")
    p.set_defaults(func=cmd_nosources)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
