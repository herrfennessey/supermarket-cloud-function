import logging
import os
import time
from pathlib import Path

from google.cloud import storage
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from categories import category_list

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

LOCAL_TMP_DIR = Path("/tmp")
MODEL_NAME = "pytorch_model.bin"
MODEL_PATH = LOCAL_TMP_DIR / MODEL_NAME

if os.path.exists(MODEL_PATH):
    logger.info("Model is in memory!")
else:
    logger.info("Have to download the model")
    download_start_time = time.time()
    prefix = "model/"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("sf-dev-conf-assets")
    model_files = bucket.list_blobs(prefix=prefix, delimiter="/")
    for blob in model_files:
        if blob.name != prefix:
            file_name = blob.name.split("/")[-1] if "/" in blob.name else blob.name
            blob.download_to_filename(LOCAL_TMP_DIR / file_name)

    logger.info(f"model download took {time.time() - download_start_time} seconds")

model_version = round(os.path.getmtime(MODEL_PATH))

model_load_start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(LOCAL_TMP_DIR)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_TMP_DIR)
pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
logger.info(f"model load time = {time.time() - model_load_start_time} seconds")


def inference(request):
    request_json = request.get_json()
    items = []
    if request.args and "item" in request.args:
        items.extend(request.args.getlist("item"))
    elif request_json and "items" in request_json:
        request_item = request_json["items"]
        if isinstance(request_item, list):
            items.extend(request_item)
        else:
            items = [request_item]
    else:
        return f"To classify multiple items, please POST a json request with your items in a JSON array with the key 'items'.\n" \
               f"To classify a single item, please use GET with a url parameter, i.e. ?item=foo"

    inference_start_time = time.time()
    label = pipeline(items)
    inference_time = (time.time() - inference_start_time) * 1000  # Convert to milliseconds
    logger.info(f"inference time = {inference_time} milliseconds")

    for label_dict in label:
        item_label = label_dict.pop("label")
        label_dict["category"] = category_list.get(int(item_label.split("_")[1]), "Unknown Category")

    item_response = dict(zip(items, label))

    return {
        "response": item_response,
        "version": model_version,
        "took": round(inference_time),
    }
