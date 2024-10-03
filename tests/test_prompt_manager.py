import unittest
from mockito import mock, when, verify
from prompts import PromptManager, PromptType  # Replace with your actual module


class TestPromptManager(unittest.TestCase):

    def setUp(self):
        self.prompt_manager = PromptManager()

    def test_get_prompt_success(self):
        # Mock the return value of _load_prompts
        mocked_prompts = {"template": "Extract product details from the flyer."}
        when(self.prompt_manager )._load_prompts(PromptType.extract_product).thenReturn(mocked_prompts)
        prompt = self.prompt_manager.get_prompt(PromptType.extract_product)
        self.assertEqual(prompt, "Extract product details from the flyer.")

        prompt = self.prompt_manager.get_prompt(PromptType.extract_product)
        self.assertEqual(prompt, "Extract product details from the flyer.")
        verify(self.prompt_manager, times=1 )._load_prompts(PromptType.extract_product)

    def test_get_prompt_no_template(self):
        mocked_prompts = {"template": ""}

        when(self.prompt_manager)._load_prompts(PromptType.extract_product).thenReturn(mocked_prompts)
        prompt = self.prompt_manager.get_prompt(PromptType.extract_product)
        self.assertIsNone(prompt)

