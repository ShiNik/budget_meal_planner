from enum import StrEnum, auto

class PromptType(StrEnum):
      EXTRACT_PRODUCT= auto()
      COLOR_CODED_EVENT_TEXT_BACKGROUND_WITH_TIME = auto()
      COLOR_CODED_EVENT_TEXT_BACKGROUND_WITH_TIME_LEGEND = auto()
      DayColorCodedWithleggend = auto()

prompt_extract_product = """Analyze the following image of a flyer. The flyer contains both English and French text about various products. 
        Please extract the information for each product, including:
        - The product name or brand logo.
        - The price of the product.
        - Any promotions, discounts, or special offers associated with the product.
        - Group all the related information for each product together.
        Ensure that both English and French details are included for each product, and clearly indicate the product's price and any promotional offers."""



def get_prompt(prompt_type:PromptType):
    if PromptType.EXTRACT_PRODUCT == prompt_type:
        return prompt_extract_product
    return None