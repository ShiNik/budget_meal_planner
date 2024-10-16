import re

#TODO: remove this code when working on All the prompts should have formated output
def extract_product_info(flyer_text):
    product_info = {
        'chicken': [],
        'beef': [],
        'pork': [],
        'fish': [],
        'vegetable': [],
    }

    product_pattern = r"\*\*(.*?)\*\*"
    price_pattern = r"\*\*Price:\*\* ([\d\.\$]+.*)"
    promo_pattern = r"\*\*Promotions:\*\* (.*?)\n"
    category_pattern = r"\*\*Category:\*\* (.*?)\n"

    # Split the flyer text by products
    products = flyer_text.split("###")[1:]

    for product in products:
        name_match = re.search(product_pattern, product)
        name = name_match.group(1) if name_match else "Not specified"
        name = name.replace("Product:", "").strip().lower()

        price_match = re.search(price_pattern, product)
        price = price_match.group(1) if price_match else "Not specified"

        promo_match = re.search(promo_pattern, product)
        promo = promo_match.group(1) if promo_match else "None"

        category_match = re.search(category_pattern, product)
        category = category_match.group(1) if category_match else "Not specified"
        category =  category.lower()

        if category in product_info:
            product_info[category].append({
                "Name": name,
                "Price": price,
                "Promotions": promo,
                "Category": category
            })

    return product_info




