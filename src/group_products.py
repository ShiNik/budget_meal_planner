def generate_product_group() -> list[tuple[str, str]]:
    #TODO: Implement the product grouping feature. Currently, it always returns a fixed set of product groups.
    return [
        ("Andouille sausage, chipotle chile pepper and chicken broth", ""),
        ("fenugreek, chicken, and onion", ""),
        ("Apricot", ""),
        ("chicken thighs", ""),
        (
            "chicken broth",
            " The recipe should have a total cooking time of less than 40 minutes and"
            " total calorie should not exceed 210.",
        ),
        (
            "sweet potato",
            " The recipe should have a total cooking time of less than 40 minutes.",
        ),
    ]
