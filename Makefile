PY := venv/Scripts/python.exe

.PHONY: setup data eda train eval notebooks test serve all

setup:            ## create the Python 3.11 environment and install dependencies
	uv venv --python 3.11 venv
	uv pip install --python $(PY) -r requirements.txt

data:             ## download the raw Criteo pool into data/raw/
	$(PY) scripts/download_data.py

eda:              ## build + execute notebook 01 (in-depth EDA)
	$(PY) scripts/build_nb01_eda.py

train:            ## build + execute notebook 02 (model training)
	$(PY) scripts/build_nb02_training.py

eval:             ## build + execute notebook 03 (evaluation)
	$(PY) scripts/build_nb03_evaluation.py

notebooks: eda train eval

test:
	$(PY) -m pytest -q

serve:            ## run the prediction API on http://127.0.0.1:8000
	$(PY) -m uvicorn ctr.service:app --app-dir src --port 8000

all: data notebooks test
