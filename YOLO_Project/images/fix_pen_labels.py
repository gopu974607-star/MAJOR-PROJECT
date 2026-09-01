from pathlib import Path

# Pen label files
pen_train = Path("C:/Users/diyad/OneDrive/Desktop/YOLO_Project/runs/pen_dataset/train/labels")
pen_valid = Path("C:/Users/diyad/OneDrive/Desktop/YOLO_Project/runs/pen_dataset/valid/labels")

# Final combined label folders
final_train = Path("C:/Users/diyad/OneDrive/Desktop/YOLO_Project/runs/final_dataset/train/labels")
final_valid = Path("C:/Users/diyad/OneDrive/Desktop/YOLO_Project/runs/final_dataset/valid/labels")

def change_labels(source_folder, destination_folder):
    for source_file in source_folder.glob("*.txt"):
        destination_file = destination_folder / source_file.name

        if destination_file.exists():
            lines = destination_file.read_text().splitlines()
            new_lines = []

            for line in lines:
                parts = line.split()
                if parts and parts[0] == "0":
                    parts[0] = "15"
                new_lines.append(" ".join(parts))

            destination_file.write_text("\n".join(new_lines))

change_labels(pen_train, final_train)
change_labels(pen_valid, final_valid)

print("Pen labels successfully changed from class 0 to class 15!")