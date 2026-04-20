import argparse
import logging
from pathlib import Path

from malaya.torch_model.t5 import T5ForTokenClassification
from malaya_boilerplate.huggingface import download_files
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    description = (
        "Downloads the pre-trained models and tokenizers that is used in testing"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("cache_directory", type=Path, help="Directory to save the models to.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cache_directory = args.cache_directory
    assert isinstance(cache_directory, Path)

    models = ["mesolitica/stem-lstm-512", "mesolitica/pos-t5-small-standard-bahasa-cased"]
    for model in models:
        
        logger.info(f"Caching model; {model} and tokenizer too: {cache_directory}")
        if model == "mesolitica/pos-t5-small-standard-bahasa-cased":
            T5ForTokenClassification.from_pretrained(model, cache_dir=cache_directory)
        else:
            s3_file = {'model': 'model.pt'}
            path = download_files(model, s3_file)
        AutoTokenizer.from_pretrained(model, cache_dir=cache_directory)
        logger.info(f"Download; {model}")