import csv
import os
import requests
from urllib.parse import urlparse

# === CONFIGURATION ===
csv_file = 'c:/projects/shopify/export.csv'  # Your CSV file
url_column_name = 'Image Src'
product_column_name = 'Handle'
base_output_folder = 'c:/projects/shopify/downloaded_images'

# === SETUP BASE OUTPUT DIRECTORY ===
os.makedirs(base_output_folder, exist_ok=True)

# === READ CSV AND DOWNLOAD IMAGES ===
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader, start=1):
        url = row.get(url_column_name)
        handle = row.get(product_column_name, 'unknown-product').strip()

        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                # Create product-specific folder
                product_folder = os.path.join(base_output_folder, handle)
                os.makedirs(product_folder, exist_ok=True)

                # Get file name from URL
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path)
                filepath = os.path.join(product_folder, filename)

                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)

                print(f"[{i}] Saved to: {filepath}")

            except Exception as e:
                print(f"[{i}] Failed to download {url}: {e}")
        else:
            print(f"[{i}] No image URL in row.")
