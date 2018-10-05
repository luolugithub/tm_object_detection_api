import os
import sys

target_dir = sys.argv[1]

count = 0
for f in sorted(os.listdir(target_dir)):
    if count % 2 == 0:
        os.remove(os.path.join(target_dir, f))
    else:
        pass
    count += 1
