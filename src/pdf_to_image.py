import asyncio
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from tqdm.asyncio import tqdm

from logger import get_logger

recipes_logger = get_logger("recipes")


async def load_page_and_convert_to_image(pdf_document: fitz.Document, output_folder: Path, page_num: int) -> None:
    # Load the page asynchronously using a thread pool
    page = await asyncio.to_thread(pdf_document.load_page, page_num)

    # Get the page's pixmap (image)
    pix = page.get_pixmap()

    # Convert the pixmap to a PIL Image
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image asynchronously using a thread pool
    image_path = f"{output_folder}/page_{page_num + 1}.png"
    await asyncio.to_thread(image.save, image_path)  # Offload saving to a separate thread
    recipes_logger.info(f"Saved page {page_num + 1} as {image_path}")


async def convert_pdf_to_images(*, pdf_path: Path, output_folder: Path) -> None:
    # Open the PDF file asynchronously (to avoid blocking)
    pdf_document = await asyncio.to_thread(fitz.open, pdf_path)

    # Create tasks for each page to load and convert asynchronously

    tasks = [
        load_page_and_convert_to_image(pdf_document, output_folder, page_num)
        for page_num in tqdm(range(len(pdf_document)), desc=f"Processing {pdf_path}")
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    recipes_logger.info("All pages converted to images.")
