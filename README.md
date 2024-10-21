[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/release/python-3123/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Coverage](coverage.svg)
[![Dependabot](https://badgen.net/badge/Dependabot/enabled/orange?icon=dependabot)](https://dependabot.com/)
[![Actions status](https://github.com/astral-sh/ruff/workflows/CI/badge.svg)](https://github.com/ShiNik/python-cicd-demo/actions)
[![image](https://img.shields.io/pypi/l/ruff.svg)](https://github.com/ShiNik/python-cicd-demo/blob/main/LICENSE)


## Step 1: Install python:
sudo apt-get install python3.12.3

## Step 2: install poetry:
curl -sSL https://install.python-poetry.org | python3 -

## Step 3: Set Up the Virtual Environment
1. Create a virtual environment by typing: `poetry install`
2. poetry run pre-commit install --hook-type commit-msg

# run pre-commit hook
by typing: `pre-commit run --all-files`

# coverage: Run pytest test coverage
```
poetry run coverage run && poetry run coverage report && poetry run coverage xml && poetry run coverage-badge -f -o coverage.svg
```

## cookbooks
Download the 2 cookbooks and copy them into project_root/data/recipes/
- [2022CookingAroundtheWorldCookbook.pdf](https://www.nutrition.va.gov/docs/UpdatedPatientEd/2022CookingAroundtheWorldCookbook.pdf)
- [cookbook.pdf](https://foodhero.org/sites/foodhero-prod/files/health-tools/cookbook.pdf)

##  Install hooks in the project
` pre-commit install --hook-type commit-msg`

## Run hooks manually on all files
- ` pre-commit run --all-files`
- ` pre-commit run <hook-id>`
- `pre-commit run --files file1.py file2.py`

For detailed information, please refer to the [Task and Improvement Tracker](https://github.com/ShiNik/budget_meal_planner/wiki),
where you can find an overview of ongoing tasks, completed improvements, and project milestones.

## Kafka
sudo docker run --name zookeeper  --network kafka-net -p 2181:2181  zookeeper
sudo docker run -p 9092:9092 --name kafka  --network kafka-net -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 confluentinc/cp-kafka
` docker-compose up -d`



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

Huggingface
https://medium.com/@scholarly360/langchain-huggingface-complete-guide-on-colab-dfafe04fe661

AI Agents RAG With LangGraph 
https://www.youtube.com/watch?v=N1FM-PcVXNA&ab_channel=KrishNaik