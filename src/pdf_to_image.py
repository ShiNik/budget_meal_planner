import fitz  # PyMuPDF
from PIL import Image
from logger import get_logger

recipes_logger = get_logger("recipes")


def convert_pdf_to_images(*, pdf_path: str, output_folder: str) -> None:
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)

        # Get the page's pixmap (image)
        pix = page.get_pixmap()

        # Convert the pixmap to a PIL Image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save the image
        image_path = f"{output_folder}/page_{page_num + 1}.png"
        image.save(image_path)
        recipes_logger.info(f"Saved page {page_num + 1} as {image_path}")

    recipes_logger.info("All pages converted to images.")
