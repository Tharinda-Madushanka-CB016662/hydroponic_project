import json
import os
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


NUMBER_OF_PLANTS = 12

CONFIG_DIR = Path("/hydroponic_project/config")
CROP_DIR = Path("/hydroponic_project/crops")

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)


if len(sys.argv) < 2:
    print(
        "Usage: python3 select_rois_tk.py "
        "/hydroponic_project/images/reference.jpg"
    )
    sys.exit(1)


IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.exists():
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)


original_image = Image.open(IMAGE_PATH).convert("RGB")

original_width, original_height = original_image.size


# ---------------------------------------------------------
# Create GUI
# ---------------------------------------------------------

root = tk.Tk()
root.title("Hydroponic Plant ROI Selector")


# Fit the image to the screen if necessary.
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

max_display_width = int(screen_width * 0.90)
max_display_height = int(screen_height * 0.80)

scale = min(
    max_display_width / original_width,
    max_display_height / original_height,
    1.0
)

display_width = int(original_width * scale)
display_height = int(original_height * scale)

display_image = original_image.resize(
    (display_width, display_height),
    Image.Resampling.LANCZOS
)

photo = ImageTk.PhotoImage(display_image)


instruction = tk.Label(
    root,
    text=(
        "Draw rectangles in order P01 → P12.  "
        "Left-to-right, top-to-bottom.  "
        "After 12 plants are selected, press SAVE."
    ),
    font=("Arial", 12)
)
instruction.pack(pady=5)


status = tk.Label(
    root,
    text="Next plant: P01",
    font=("Arial", 12, "bold")
)
status.pack(pady=3)


canvas = tk.Canvas(
    root,
    width=display_width,
    height=display_height,
    cursor="cross"
)

canvas.pack()

canvas.create_image(
    0,
    0,
    anchor=tk.NW,
    image=photo
)


start_x = None
start_y = None
current_rectangle = None

rois = []


def mouse_down(event):
    global start_x, start_y, current_rectangle

    if len(rois) >= NUMBER_OF_PLANTS:
        return

    start_x = event.x
    start_y = event.y

    current_rectangle = canvas.create_rectangle(
        start_x,
        start_y,
        start_x,
        start_y,
        outline="red",
        width=3
    )


def mouse_move(event):
    if current_rectangle is None:
        return

    canvas.coords(
        current_rectangle,
        start_x,
        start_y,
        event.x,
        event.y
    )


def mouse_up(event):
    global current_rectangle

    if current_rectangle is None:
        return

    x1 = min(start_x, event.x)
    y1 = min(start_y, event.y)

    x2 = max(start_x, event.x)
    y2 = max(start_y, event.y)

    if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
        canvas.delete(current_rectangle)
        current_rectangle = None
        return

    # Convert display coordinates back to original image coordinates.
    original_x1 = int(x1 / scale)
    original_y1 = int(y1 / scale)

    original_x2 = int(x2 / scale)
    original_y2 = int(y2 / scale)

    plant_number = len(rois) + 1

    roi = {
        "plant_id": f"P{plant_number:02d}",
        "x": original_x1,
        "y": original_y1,
        "w": original_x2 - original_x1,
        "h": original_y2 - original_y1
    }

    rois.append(roi)

    # Put plant ID on selected rectangle.
    canvas.create_text(
        x1 + 5,
        y1 + 5,
        anchor=tk.NW,
        text=roi["plant_id"],
        fill="red",
        font=("Arial", 14, "bold")
    )

    current_rectangle = None

    if len(rois) < NUMBER_OF_PLANTS:
        status.config(
            text=f"Next plant: P{len(rois) + 1:02d}"
        )
    else:
        status.config(
            text="All 12 plants selected. Press SAVE."
        )


def undo_last():
    global rois

    if not rois:
        return

    rois.pop()

    # easiest reliable redraw
    canvas.delete("all")

    canvas.create_image(
        0,
        0,
        anchor=tk.NW,
        image=photo
    )

    for roi in rois:
        x1 = int(roi["x"] * scale)
        y1 = int(roi["y"] * scale)

        x2 = int((roi["x"] + roi["w"]) * scale)
        y2 = int((roi["y"] + roi["h"]) * scale)

        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline="red",
            width=3
        )

        canvas.create_text(
            x1 + 5,
            y1 + 5,
            anchor=tk.NW,
            text=roi["plant_id"],
            fill="red",
            font=("Arial", 14, "bold")
        )

    status.config(
        text=f"Next plant: P{len(rois) + 1:02d}"
    )


def save_rois():
    if len(rois) != NUMBER_OF_PLANTS:
        status.config(
            text=(
                f"Please select all {NUMBER_OF_PLANTS} plants. "
                f"Currently selected: {len(rois)}"
            )
        )
        return

    config_file = CONFIG_DIR / "plant_rois.json"

    with open(config_file, "w") as file:
        json.dump(rois, file, indent=4)

    # Save plant crops immediately.
    for roi in rois:

        x1 = roi["x"]
        y1 = roi["y"]

        x2 = x1 + roi["w"]
        y2 = y1 + roi["h"]

        crop = original_image.crop(
            (x1, y1, x2, y2)
        )

        original_crop = (
            CROP_DIR /
            f'{roi["plant_id"]}.jpg'
        )

        crop.save(
            original_crop,
            quality=95
        )

        # MobileNetV2-sized copy.
        resized = crop.resize(
            (224, 224),
            Image.Resampling.LANCZOS
        )

        resized_path = (
            CROP_DIR /
            f'{roi["plant_id"]}_224.jpg'
        )

        resized.save(
            resized_path,
            quality=95
        )

    print()
    print("ROI configuration saved:")
    print(config_file)

    print()
    print("Plant crops saved:")
    print(CROP_DIR)

    root.destroy()


canvas.bind("<ButtonPress-1>", mouse_down)
canvas.bind("<B1-Motion>", mouse_move)
canvas.bind("<ButtonRelease-1>", mouse_up)


button_frame = tk.Frame(root)
button_frame.pack(pady=8)


undo_button = tk.Button(
    button_frame,
    text="UNDO LAST",
    command=undo_last,
    width=15
)

undo_button.pack(
    side=tk.LEFT,
    padx=10
)


save_button = tk.Button(
    button_frame,
    text="SAVE 12 PLANTS",
    command=save_rois,
    width=20
)

save_button.pack(
    side=tk.LEFT,
    padx=10
)


root.mainloop()
