# stage_1.py
# Move ~1/3 of training_data (per class) into prior_data.
# The moved files disappear from training_data (because move).
#
# Folder structure expected:
#   training_data/<class_name>/*
#   prior_data/ (will be created) /<class_name>/*
#
# Run:
#   python stage_1.py
#
# Optional: change SEED and PRIOR_FRAC below.

import os
import random
import shutil

SEED = 42
PRIOR_FRAC = 1.0 / 3.0

TRAIN_DIR = "training_data"
PRIOR_DIR = "prior_data"

# Common image extensions (add more if you want)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTS)


def main():
    random.seed(SEED)

    if not os.path.isdir(TRAIN_DIR):
        raise RuntimeError(f"Can't find '{TRAIN_DIR}' folder.")

    # Create prior_data if missing
    os.makedirs(PRIOR_DIR, exist_ok=True)

    # Each class is a subfolder inside training_data
    class_names = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    if not class_names:
        raise RuntimeError(f"No class folders found inside '{TRAIN_DIR}'.")

    print("Classes found:", ", ".join(sorted(class_names)))
    print(f"Moving about {PRIOR_FRAC:.3f} of each class into '{PRIOR_DIR}' (seed={SEED})")
    print()

    total_moved = 0

    for cls in sorted(class_names):
        src_cls_dir = os.path.join(TRAIN_DIR, cls)
        dst_cls_dir = os.path.join(PRIOR_DIR, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # List images
        files = [f for f in os.listdir(src_cls_dir) if is_image(f)]
        files.sort()

        if len(files) == 0:
            print(f"[{cls}] No images found. Skipping.")
            continue

        random.shuffle(files)

        n_total = len(files)
        n_move = int(n_total * PRIOR_FRAC)

        # Ensure we move at least 1 file if there are enough files
        if n_total >= 3 and n_move == 0:
            n_move = 1

        move_list = files[:n_move]

        for fname in move_list:
            src_path = os.path.join(src_cls_dir, fname)
            dst_path = os.path.join(dst_cls_dir, fname)

            # If a file with the same name already exists in prior_data, rename it
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(fname)
                k = 1
                while True:
                    new_name = f"{base}__dup{k}{ext}"
                    new_dst = os.path.join(dst_cls_dir, new_name)
                    if not os.path.exists(new_dst):
                        dst_path = new_dst
                        break
                    k += 1

            shutil.move(src_path, dst_path)
            total_moved += 1

        print(f"[{cls}] total={n_total}  moved={n_move}  left={n_total - n_move}")

    print()
    print(f"Done. Moved {total_moved} files into '{PRIOR_DIR}'.")


if __name__ == "__main__":
    main()
