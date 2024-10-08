## Prerequisites
- ` python 3.12`
- ` pip`
- ` venv`
## cookbooks
Download the 2 cookbook and copy them into project_roo/data/recipes/
- [2022CookingAroundtheWorldCookbook.pdf](https://www.nutrition.va.gov/docs/UpdatedPatientEd/2022CookingAroundtheWorldCookbook.pdf)
- [cookbook.pdf](https://foodhero.org/sites/foodhero-prod/files/health-tools/cookbook.pdf)

##  Install hooks in the project
` pre-commit install --hook-type commit-msg`

## Run hooks manually on all files
- ` pre-commit run --all-files`
- ` pre-commit run <hook-id>`
- `pre-commit run --files file1.py file2.py`


## Download Ollama and Choose a Model

1. Visit [Ollama](https://ollama.com/) to download Ollama.
2. Go to the [Ollama Library](https://ollama.com/library) to choose a model.

##  Use Llama3

For using llama3, follow these steps:

1. Visit the [Llama3 Blog](https://ollama.com/blog/llama3) for usage instructions.
2. To download llama3, open a terminal and type: `ollama pull llama3`.
3. To check the list of models, in terminal type: `ollama list`.
4. To run the model, type: `ollama run llama3`.
5. If you would like to see the output of the model for debugging, you can run ollama as a server by typing: `ollama serve`.

useful links
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/code
https://www.kaggle.com/code/engyyyy/recipe-recommendation-system
https://www.kaggle.com/code/stevenadisantoso/food-recommendation-using-tfrs
https://www.youtube.com/watch?v=jeCYqCEhqd8

https://www.youtube.com/@fastandsimpledevelopment/videos
https://www.youtube.com/watch?v=V0cEMA_D8jw
https://drive.google.com/drive/folders/1A7xufWkxwzgt40rl-vaEwC0xk2sp8bvJ


Chat With Multiple PDF Documents With Langchain And Google Gemini Pro
https://www.youtube.com/watch?v=uus5eLz6smA&list=PLZoTAELRMXVORE4VF7WQ_fAl0L1Gljtar&index=16
https://github.com/krishnaik06/Complete-Langchain-Tutorials/blob/main/chatmultipledocuments/chatpdf1.py