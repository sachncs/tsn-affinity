"""Unified CLI entry point for TSN-Affinity."""

import sys


def main() -> None:
    """Print help or dispatch to subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python -m tsn_affinity.cli <command> [options]")
        print("Commands: benchmark, atari, panda")
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "benchmark":
        from tsn_affinity.cli.benchmark import main as benchmark_main

        sys.argv = [sys.argv[0]] + args
        benchmark_main()
    elif command == "atari":
        from tsn_affinity.cli.atari import main as atari_main

        sys.argv = [sys.argv[0]] + args
        atari_main()
    elif command == "panda":
        from tsn_affinity.cli.panda import main as panda_main

        sys.argv = [sys.argv[0]] + args
        panda_main()
    else:
        print(f"Unknown command: {command}")
        print("Commands: benchmark, atari, panda")
        sys.exit(1)


if __name__ == "__main__":
    main()
