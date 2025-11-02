#!/usr/bin/env python3

import argparse
from lib import semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    model_parser = subparsers.add_parser("verify", help="Create and Verify Model")

    args = parser.parse_args()

            

    match args.command:
        case "verify":
            verify()
        case _:
            parser.print_help()

def verify():
    model = semantic_search.verify_model()

if __name__ == "__main__":
    main()
