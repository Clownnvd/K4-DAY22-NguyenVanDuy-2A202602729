"""Render a readable excerpt of the eight fixed NB4 comparisons.

The raw model outputs and judge verdicts remain in data/eval/. This image is
only a reading aid; it deliberately shows an excerpt of each answer.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "eval" / "side_by_side.jsonl"
JUDGED = ROOT / "data" / "eval" / "judge_results_rm.json"
OUTPUT = ROOT / "submission" / "screenshots" / "04-side-by-side-table-readable.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    choices = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in choices:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("A font with Vietnamese glyphs is required")


def excerpt(text: str, width: int = 58, limit: int = 175) -> list[str]:
    clean = re.sub(r"</?tool_call>", "", text)
    clean = " ".join(clean.split())
    if len(clean) > limit:
        clean = clean[:limit].rsplit(" ", 1)[0] + "…"
    return textwrap.wrap(clean, width=width, break_long_words=False)[:4]


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()][:8]
    verdicts = {row["id"]: row["winner"] for row in json.loads(JUDGED.read_text(encoding="utf-8"))["records"][:8]}
    width, height = 3200, 2060
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title, head, body, small = font(58, True), font(36, True), font(32), font(26)
    ink, muted, line = "#16253a", "#536279", "#dbe2e9"

    draw.text((90, 58), "So sánh 8 câu hỏi cố định", font=title, fill=ink)
    draw.text((90, 136), "SFT và SFT + DPO - trích câu trả lời, không thay đổi dữ liệu gốc", font=body, fill=muted)
    header_y, row_h = 226, 206
    draw.rounded_rectangle((72, header_y, width - 72, header_y + 74), radius=15, fill="#203d63")
    for x, label in [(100, "Câu hỏi / kết quả"), (670, "SFT"), (1910, "SFT + DPO")]:
        draw.text((x, header_y + 17), label, font=head, fill="white")

    for index, row in enumerate(rows):
        top = header_y + 74 + index * row_h
        bottom = top + row_h
        if index % 2 == 0:
            draw.rectangle((72, top, width - 72, bottom), fill="#f7f9fc")
        draw.line((72, bottom, width - 72, bottom), fill=line, width=2)
        result = {"dpo": "DPO thắng", "sft": "SFT thắng", "tie": "Hòa"}.get(verdicts.get(row["id"]), "Chưa chấm")
        draw.text((100, top + 20), f"{row['id'].upper()}  {result}", font=head, fill=ink)
        prompt = textwrap.wrap(row["prompt"], width=30, break_long_words=False)[:3]
        for offset, part in enumerate(prompt):
            draw.text((100, top + 75 + offset * 34), part, font=small, fill=muted)
        for x, key in [(670, "sft"), (1910, "dpo")]:
            for offset, part in enumerate(excerpt(row[key])):
                draw.text((x, top + 22 + offset * 42), part, font=body, fill=ink)
    draw.text((90, height - 95), "Mỗi ô chỉ hiện đoạn đầu. Xem toàn văn và token tool_call trong data/eval/side_by_side.jsonl.", font=small, fill=muted)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
