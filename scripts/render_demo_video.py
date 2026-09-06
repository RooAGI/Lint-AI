#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1152, 648
FPS = 8
BG = (15, 17, 21)
BAR = (31, 34, 41)
FG = (229, 231, 235)
MUTED = (156, 163, 175)
ACCENT = (125, 211, 252)
GREEN = (134, 239, 172)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


MONO_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
MONO = font(MONO_PATH, 22)
BOLD = font(BOLD_PATH, 22)
SMALL = font(MONO_PATH, 17)


def wrap_line(line, width=82):
    if len(line) <= width:
        return [line]
    indent = "  " if not line.startswith("$") else ""
    out = []
    rest = line
    while len(rest) > width:
        cut = rest.rfind(" ", 0, width)
        if cut < width // 2:
            cut = width
        out.append(rest[:cut])
        rest = indent + rest[cut:].lstrip()
    out.append(rest)
    return out


def draw_frame(lines, footer):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((24, 24, W - 24, H - 24), radius=18, fill=(20, 23, 29))
    d.rounded_rectangle((24, 24, W - 24, 76), radius=18, fill=BAR)
    d.rectangle((24, 58, W - 24, 76), fill=BAR)
    for x, c in [(50, (248, 113, 113)), (76, (251, 191, 36)), (102, (74, 222, 128))]:
        d.ellipse((x - 7, 49 - 7, x + 7, 49 + 7), fill=c)
    d.text((132, 39), "Lint-AI • reproducible current-state demo", font=SMALL, fill=MUTED)

    y = 98
    visible = []
    for raw in lines:
        visible.extend(wrap_line(raw))
    visible = visible[-21:]
    for line in visible:
        color = FG
        f = MONO
        if line.startswith("$"):
            color, f = ACCENT, BOLD
        elif '"semantic_status": "current"' in line or "retry attempts: 2" in line:
            color = GREEN
        elif "decision-a.md" in line and "supersedes" not in line and not line.startswith("$"):
            color = MUTED
        d.text((48, y), line, font=f, fill=color)
        y += 25
    d.text((48, H - 48), footer, font=SMALL, fill=MUTED)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--gif", required=True)
    ap.add_argument("--mp4", required=True)
    args = ap.parse_args()

    lines = Path(args.transcript).read_text().splitlines()
    frames = []
    shown = []
    for line in lines:
        if line.startswith("$"):
            for n in range(1, 7):
                partial = line[: max(1, int(len(line) * n / 6))]
                frames.append(draw_frame(shown + [partial + ("▌" if n < 6 else "")], "Generated from the actual lint-ai binary and captured JSON output"))
            shown.append(line)
        else:
            shown.append(line)
            frames.extend([draw_frame(shown, "Generated from the actual lint-ai binary and captured JSON output")] * 3)

    frames.extend([draw_frame(shown, "Relevant context ≠ current context • github.com/RooAGI/Lint-AI")] * 18)
    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, frame in enumerate(frames):
            frame.save(td / f"frame-{i:04d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "frame-%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
