## Settings things up on gradient

1. Download files from kaggle to `raw`
2. Deploy them to gradient

   TODO

3. Start an experiment
    gradient experiments run singlenode --optionsFile gradient.yml 

# Unsorted
## Running locally

    virtualenv venv
    source venv/bin/activate
    cat ../../docker-kaggle-deeplearning-all/requirements.txt | sed 's/+cu101//g' | pip install -r /dev/stdin
    python3 catalyst_pipeline.py
