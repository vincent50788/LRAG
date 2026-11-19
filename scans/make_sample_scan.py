"""Make the sample scan: scans/starriver-tech-overseas-travel-guide.pdf (4 pages).

The script draws the text as pictures. Thus the PDF has no text layer, like a real scan.
Then it adds a small rotation, blur, and noise, like a real scanner does.
The text below is also the right answer for OCR. To check OCR, compare it with data/starriver-tech-overseas-travel-guide.md.
The guide fills the gap in the handbook ("Different limits apply to overseas regions").

    .venv/bin/python scans/make_sample_scan.py
"""
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent / "starriver-tech-overseas-travel-guide.pdf"
DPI = 200
W, H = int(8.27 * DPI), int(11.69 * DPI)  # A4 paper
MARGIN = 170
TILT = (0.9, -0.7, 1.2, -1.0)  # Degrees. Each page sits a little crooked on the scanner.

PAGES = [
    ("1. Booking and Approval", [
        "Every overseas trip needs a travel request form. Your department manager and your department vice president must both approve it before you book anything.",
        "Book flights through the company travel desk, Skyline Travel, at least 14 days before departure. A booking made later needs a written reason from your manager.",
        "Fly economy class. For a flight of 6 hours or more, you may use premium economy. Business class is not allowed, unless the CEO approves it in writing.",
        "Your passport must be valid for at least 6 months after the day you return. You are responsible for your visa. The company pays the visa fee, but it does not pay for a passport.",
    ]),
    ("2. Hotels and Daily Allowance", [
        "The hotel limit for each night depends on the region. Asia: NT$4,500. Europe and North America: NT$7,000. All other regions: NT$5,500. The company pays only for the room. Laundry and the minibar are your own cost.",
        "The daily meal allowance is NT$1,200 in Asia, NT$1,800 in Europe, and NT$2,000 in North America. You do not need receipts for the allowance. If your hotel includes breakfast, subtract NT$300 from the allowance for that day.",
        "Use public transport or the airport train first. A taxi is allowed from the airport to the hotel when you carry a large bag or arrive after 22:00.",
    ]),
    ("3. Insurance and Emergencies", [
        "The company buys travel insurance for every overseas trip. You do not need to apply. It pays up to NT$5,000,000 for an accident and up to NT$300,000 for medical costs. Keep every medical receipt.",
        "In an emergency, call the Starriver Travel Hotline at +886-2-5550-0199. It works 24 hours every day.",
        "If you lose your passport, report it to the local police first. Then contact the Taiwan representative office in that country, and call HR within 24 hours. If your flight is cancelled, call Skyline Travel. The company pays the rebooking fee.",
    ]),
    ("4. Cash Advance and Coming Home", [
        "You may ask for a cash advance of up to NT$30,000 for an overseas trip. Send the request to the Finance Department at least 5 working days before you leave. Return any unused cash within 5 days after you come back.",
        "Write a short trip report within 7 days after you return. Send it to your manager and to the Finance Department.",
        "After a flight of 10 hours or more, you may start work at 13:00 on the next working day. You do not need to take leave for this.",
    ]),
]


def wrap(draw, text, font, width):
    """Break the text into lines that fit the width."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def draw_page(heading, paragraphs):
    """Draw one clean page: a running header, a heading, and the paragraphs."""
    page = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(page)
    head_font, body_font, small_font = (ImageFont.load_default(size=s) for s in (58, 36, 26))
    draw.text((MARGIN, 110), "Starriver Tech  |  Overseas Travel Guide", font=small_font, fill=110)
    draw.text((MARGIN, 260), heading, font=head_font, fill=15)
    y = 390
    for paragraph in paragraphs:
        for line in wrap(draw, paragraph, body_font, W - 2 * MARGIN):
            draw.text((MARGIN, y), line, font=body_font, fill=25)
            y += 58
        y += 40
    return page


def scan_effect(page, angle):
    """Make the clean page look scanned: soft edges, crooked, gray paper, and noise."""
    page = page.filter(ImageFilter.GaussianBlur(0.8))
    page = page.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    page = ImageChops.multiply(page, Image.new("L", page.size, 240))  # Paper is not pure white.
    noise = Image.effect_noise(page.size, 6)  # Gray noise around 128. A higher number gives more noise and a larger file.
    return ImageChops.add(page, noise, 1.0, -128)  # Add the noise to the page. Subtract the 128.


if __name__ == "__main__":
    pages = [scan_effect(draw_page(*content), angle) for content, angle in zip(PAGES, TILT)]
    # Real scanned PDFs are JPEG files too. Quality 50 keeps this file near 1 MB.
    pages[0].save(OUT, "PDF", save_all=True, append_images=pages[1:], resolution=DPI, quality=50)
    print(f"Wrote {OUT.name}: {len(pages)} pages, {OUT.stat().st_size / 1024:.0f} KB")
