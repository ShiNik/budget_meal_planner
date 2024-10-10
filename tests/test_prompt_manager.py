import unittest
from mockito import when, verify
from prompts import PromptManager
from config import get_config
from common import TaskType

config = get_config()



class TestPromptManager(unittest.TestCase):

    def setUp(self):
        self.prompt_manager = PromptManager(config)

    def test_get_prompt_success(self):
        # Mock the return value of _load_prompts
        mocked_prompts = {"template": "Extract product details from the flyer."}
        when(self.prompt_manager )._load_prompts(TaskType.EXTRACT_PRODUCT).thenReturn(mocked_prompts)
        prompt = self.prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT)
        self.assertEqual(prompt, "Extract product details from the flyer.")

        prompt = self.prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT)
        self.assertEqual(prompt, "Extract product details from the flyer.")
        verify(self.prompt_manager, times=1 )._load_prompts(TaskType.EXTRACT_PRODUCT)

    def test_get_prompt_no_template(self):
        mocked_prompts = {"template": None}

        when(self.prompt_manager)._load_prompts(TaskType.EXTRACT_PRODUCT).thenReturn(mocked_prompts)
        prompt = self.prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT)
        self.assertIsNone(prompt)

