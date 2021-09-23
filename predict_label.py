import os
import tqdm
import argparse
import collections
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from models import SemanticMatching
from predictors import SemanticMatchingPredictor
from dataset_readers import SemanticMatchingDatasetReader
import json
import pandas as pd
import numpy as np
from allennlp.data import vocabulary
import tqdm
vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert


def main(args):
    df = pd.read_csv('/home1/CM2021/zwh/QQdata/data_sample.csv')
    answer_corpus = []
    total_seed = np.arange(len(df))
    for item in total_seed:
        temp = str(df.iloc[item, 0]).strip()
        if len(temp) > 160:
            # 截断数据
            temp = temp[0 : 150]
        answer_corpus.append(temp)
    archive = load_archive(args.output_dir)
    predictors = Predictor.from_archive(archive=archive, predictor_name="bert_semantic_matching")
    result = []
    result_logits = []
    for item in tqdm.tqdm(answer_corpus):
        temp_dict = {}
        temp_dict["sent1"] = "青年干部在实践锻炼中有什么成长"
        temp_dict["sent2"] = item
        output = predictors.predict_json(temp_dict)
        if output['predicted_labels'] == '0':
            result.append(item)
            result_logits.append(output['logits'])
    #output = predictors.predict_json({"sent1": "党的中央委员会一届任期是多久？", "sent2": "无论任何时候，都要牢牢坚持党的领导"})
    #label = output["predicted_labels"]
    #logits = output
    print("bupt")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, default="/home1/CM2021/zwh/QA/bert_semantic_matching-master/output/bert-output8",
                            help="the directory that stores training output")
    args = arg_parser.parse_args()
    main(args)