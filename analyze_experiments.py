import json
import os
import sys


def load_stats(path):
    with open(path, "r") as file:
        return json.load(file)


def print_stats(path):
    data = load_stats(path)

    print(f"\n{'=' * 70}")
    print(path)
    print("=" * 70)

    for agent, stats in data.items():
        print(f"\n{agent}")

        if not isinstance(stats, dict):
            print(stats)
            continue

        for key, value in stats.items():
            print(f"  {key:25s}: {value}")


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [
            name for name in os.listdir(".")
            if os.path.isfile(name)
            and (
                name.startswith("task2_")
                or name.startswith("task3_")
                or name.startswith("task4_")
                or name.startswith("qlearning_")
            )
        ]
        files.sort()

    for path in files:
        try:
            print_stats(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass


if __name__ == "__main__":
    main()