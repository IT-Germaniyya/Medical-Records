"""Create synthetic, non-PHI PDF evidence for a local MVP demonstration."""
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from PIL import Image, ImageDraw


def create_pdf(destination: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }),
        }),
    })
    lines = [
        "Date: 2026-01-14",
        "Diagnosis: Bronchial asthma",
        "Medication: Salbutamol 100 mcg inhaler",
        "Hb: 10.2 g/dL",
        "MCV: 74 fL",
        "Platelets: 516 x10^9/L",
        "Weight: 18.2 kg",
        "Height: 108 cm",
    ]
    commands = ["BT /F1 12 Tf 72 720 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("(", "\\(").replace(")", "\\)")
        commands.append(f"({escaped}) Tj")
        if index < len(lines) - 1:
            commands.append("0 -22 Td")
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = stream
    with destination.open("wb") as handle:
        writer.write(handle)


def create_image(destination: Path) -> None:
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 90), "Synthetic source document - not a real patient", fill="#102a43")
    draw.text((80, 160), "Prescription image retained for detailed report appendix", fill="#17324b")
    image.save(destination)


if __name__ == "__main__":
    target = Path("sample_patients/PATIENT_DEMO_0001")
    target.mkdir(parents=True, exist_ok=True)
    create_pdf(target / "CBC_lab_report.pdf")
    create_image(target / "prescription.png")
    print(target)
