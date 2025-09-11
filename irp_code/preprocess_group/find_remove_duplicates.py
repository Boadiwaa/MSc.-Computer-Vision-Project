#!/usr/bin/env python3
"""
find_remove_duplicates.py

Scan multiple folders of images, compute a perceptual hash (pHash)
for each image,and detect duplicates across folders
(or within the same folder). Optionally remove
duplicates (keeping only one copy). This script uses the `imagehash` library
for hashing and `PIL` for image processing.
Requirements:
    - Python 3.x
    - `Pillow` for image handling: `pip install Pillow`
    - `imagehash` for perceptual hashing: `pip install ImageHash`
    - `argparse` for command-line argument parsing
        (included in Python standard library)

NB: Tbis script was created with the help of GitHub Copilot.
Thoroughly edited and tested by me.
All comments, and descriptions are of my own writing.

Usage:
    python find_remove_duplicates.py --folders /path/to/folder1 \
                                        /path/to/folder2 \
                                    [--hash-size 16] [--max-distance 5] \
                                    [--verbose] \
                                    [--delete]
"""

import os
import sys
import argparse
from PIL import Image
import imagehash


def compute_phash(image_path, hash_size=16):
    """
    Open an image and compute its perceptual hash (pHash).
    - hash_size controls the resolution of the hash:
    e.g. 16 refers to a 16x16 DCT-based hash.
    Returns: imagehash.ImageHash object
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and compute pHash
            phash = imagehash.phash(img, hash_size=hash_size)
            return phash
    except Exception as e:
        print(f"Warning: Could not open or hash {image_path}: {e}")
        return None


def find_all_images(
    root_folder, exts={".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
):
    """
    Recursively walk `root_folder` and yield full paths to all files whose
    extension is in `exts` (case-insensitive).
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            _, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                yield os.path.join(dirpath, fname)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Find (and optionally delete) duplicate
        images across multiple folders
        by comparing their perceptual hashes (pHash).
        """
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="""
        One or more folders to scan. Duplicates will be reported
        across all of them.""",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="""
        Hash size for pHash (default: 16).
        Larger hash size is more precise but slower.
        """,
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5,
        help="""
        Maximum Hamming distance between two hashes to be
        considered duplicates (default: 5).
        """,
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, actually delete duplicates (keeps the first-seen copy).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information about hashing progress.",
    )
    args = parser.parse_args()

    # 1) Collect all image file paths from the listed folders
    all_image_paths = []
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"Error: {folder} is not a directory or does not exist.")
            sys.exit(1)
        for img_path in find_all_images(folder):
            all_image_paths.append(img_path)

    if len(all_image_paths) == 0:
        print("No images found in the specified folders.")
        sys.exit(0)

    # 2) Compute pHash for each image and store in a dict
    #    phash_map: { image_path: ImageHash }
    phash_map = {}
    print(
        f"Computing pHash for {len(all_image_paths)} \
              images (hash_size={args.hash_size})..."
    )
    for idx, img_path in enumerate(all_image_paths):
        # Print progress of every 100 images if verbose mode is enabled
        if args.verbose and idx % 100 == 0:
            print(f"Hashing image {idx + 1}/{len(all_image_paths)}\
                  : {img_path}")
        ph = compute_phash(img_path, hash_size=args.hash_size)
        if ph is not None:
            phash_map[img_path] = ph
    print("Done hashing.\n")

    # 3) Group images by similar hash (within max_distance)
    #    Approach: iterate through sorted list of (path, hash). For each,
    #    compare to previously-seen “canonical” hashes (reference points).
    #    If it’s close to any, i.e. within max_distance,
    #    mark as duplicate; otherwise add as a new canonical hash.
    #    canonical_hashes: list of tuples (canonical_path, canonical_phash)
    #    duplicates: list of (dup_path, dup_phash, matched_canonical_path)
    canonical_hashes = []
    duplicates = []

    print(f"Comparing hashes (max Hamming distance = {args.max_distance})...")
    for img_path, ph in phash_map.items():
        if not canonical_hashes:
            # The very first image becomes a “canonical” reference
            canonical_hashes.append((img_path, ph))
            continue

        found_dup = False
        for canon_path, canon_ph in canonical_hashes:
            dist = ph - canon_ph  # Hamming distance
            if dist <= args.max_distance:
                # Considered a duplicate of `canon_path`
                duplicates.append((img_path, ph, canon_path, dist))
                found_dup = True
                break

        if not found_dup:
            # No close match found → this becomes a new canonical hash
            canonical_hashes.append((img_path, ph))

    print(f"Found {len(duplicates)} duplicates out \
          of {len(phash_map)} images.\n")

    # 4) Report (and optionally delete) duplicates
    if duplicates:
        print(
            "Duplicate summary (format: [DUPLICATE_PATH] ->\
                  [CANONICAL_PATH], Hamming distance):\n"
        )
        for dup_path, dup_ph, canon_path, dist in duplicates:
            print(
                f"  {dup_path}\n    ↳ Duplicate of: {canon_path}\
                            (distance = {dist})\n"
            )

        if args.delete:
            print("Deleting duplicates:")
            for dup_path, _, canon_path, dist in duplicates:
                try:
                    os.remove(dup_path)
                    print(f"  Deleted {dup_path} \
                          (matched {canon_path}, dist={dist})")
                except Exception as e:
                    print(f"  ERROR deleting {dup_path}: {e}")
    else:
        print("No duplicates detected. All images appear unique.")


if __name__ == "__main__":
    main()
