"""
Run this file to convert the original raw generated data to MOT format
"""

import argparse
import glob
import shutil
import os

def extract_raw_label(text):
    lines = text.split("\n")
    cam_info = list(map(float, lines[0].split(" ")))
    cam_coords, cam_rot = cam_info[:3], cam_info[3:]
    num_people = int(lines[1])
    num_group = int(lines[2])

    line_idx = 3
    groups = []
    for group_idx in range(num_group):
        group_count = int(lines[line_idx])
        line_idx += 1
        group = []
        for person_idx in range(group_count):
            group.append(list(map(float, lines[line_idx].split(" "))))
            line_idx += 1
        groups.append(group)

    return cam_coords, cam_rot, num_people, num_group, groups



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="../data/gta_raw")
    parser.add_argument("--preprocessed_path", type=str, default="../data/gta_preprocessed")

    args = parser.parse_args()

    print(args.preprocessed_path)
    os.makedirs(args.preprocessed_path, exist_ok=True)

    for txt_path in glob.glob(f"{args.raw_path}/*.txt"):
        print(txt_path)
        filename = ".".join(txt_path.split("/")[-1].split(".")[:-1])
        shutil.copyfile(f"{args.raw_path}/{filename}.jpg", f"{args.preprocessed_path}/{filename}.jpg")

        with open(txt_path) as fin, open(f"{args.preprocessed_path}/{filename}.txt", "w") as fout:
            data = fin.read()
            cam_coords, cam_rot, num_people, num_group, groups = extract_raw_label(data)
            for group_idx, group in enumerate(groups):
                for person_idx, person in enumerate(group):
                    fout.write(f"0 {group_idx} {' '.join([str(i) for i in person])}\n")



if __name__ == "__main__":
    main()