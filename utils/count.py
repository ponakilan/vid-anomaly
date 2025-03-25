import os
import sys


def main(path: str):
    dirs = list(os.walk(path))
    min_files = 999
    max_files = 0
    for i in range(1, len(dirs)):
        min_files = min(min_files, len(dirs[i][2]))
        max_files = max(max_files, len(dirs[i][2]))
    print(min_files, max_files)

if __name__ == "__main__":
    args = sys.argv
    main(args[1])
