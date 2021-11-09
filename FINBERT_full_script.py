###################################################################################

############################# Master's Thesis #####################################
#Does sentiment from relevant news improve volatility forecast from GARCH models? -
#An application of natural language processing

###################################################################################

from __future__ import absolute_import, division, print_function
from typing import Text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import random

import pandas as pd
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
    TensorDataset)
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import numpy as np
import logging

logger = logging.getLogger(__name__)


# The classes used for data processing and convert_examples_to_features are very similar versions of the ones \
# found in Hugging Face's scripts in the transformers library. For more BERT or similar language model implementation \
# examples, we would highly recommend checking that library as well.

# Classes regarding input and data handling

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'config.json'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, agree=None):
        """
        Constructs an InputExample
        Parameters
        ----------
        guid: str
            Unique id for the examples
        text: str
            Text for the first sequence.
        label: str, optional
            Label for the example.
        agree: str, optional
            For FinBERT , inter-annotator agreement level.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.agree = agree

class InputFeatures(object):
    """
    A single set of features for the data.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, agree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.agree = agree

class DataProcessor(object):
    """Base class to read data files."""

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        return lines

class FinSentProcessor(DataProcessor):
    """
    Data processor for FinBERT.
    """

    def get_examples(self, data_dir, phase):
        """
        Get examples from the data directory.
        Parameters
        ----------
        data_dir: str
            Path for the data directory.
        phase: str
            Name of the .csv file to be loaded.
        """
        return self._create_examples(self._read_tsv(os.path.join(data_dir, (phase + ".csv"))), phase)

    def get_labels(self):
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text = line[1]
            label = line[2]
            try:
                agree = line[3]
            except:
                agree = None
            examples.append(
                InputExample(guid=guid, text=text, label=label, agree=agree))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode='classification'):
    """
    Loads a data file into a list of InputBatch's. With this function, the InputExample's are converted to features
    that can be used for the model. Text is tokenized, converted to ids and zero-padded. Labels are mapped to integers.
    Parameters
    ----------
    examples: list
        A list of InputExample's.
    label_list: list
        The list of labels.
    max_seq_length: int
        The maximum sequence length.
    tokenizer: BertTokenizer
        The tokenizer to be used.
    mode: str, optional
        The task type: 'classification' or 'regression'. Default is 'classification'
    Returns
    -------
    features: list
        A list of InputFeature's, which is an InputBatch.
    """

    if mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}
        label_map[None] = 9090

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length // 4) - 1] + tokens[
                                                          len(tokens) - (3 * max_seq_length // 4) + 1:]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding


        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if mode == 'classification':
            label_id = label_map[example.label]
        elif mode == 'regression':
            label_id = float(example.label)
        else:
            raise ValueError("The mode should either be classification or regression. You entered: " + mode)

        agree = example.agree
        mapagree = {'0.5': 1, '0.66': 2, '0.75': 3, '1.0': 4}
        try:
            agree = mapagree[agree]
        except:
            agree = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          agree=agree))
    return features

def accuracy(out, labels):
    """In mathematics, the arguments of the maxima (abbreviated arg max or argmax) 
    are the points, or elements, of the domain of some function at which the function 
    values are maximized.[note 1] In contrast to global maxima, which refers to the 
    largest outputs of a function, arg max refers to the inputs, or arguments, 
    at which the function outputs are as large as possible."""
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    The softmax function, also known as softargmax[1]:184 or 
    normalized exponential function,[2]:198 is a generalization 
    of the logistic function to multiple dimensions
    but after applying softmax, each component will be in the interval (0,1)"""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]

def chunks(l, n):
    """
    Simple utility function to split a list into fixed-length chunks.
    Parameters
    ----------
    l: list
        given list
    n: int
        length of the sequence
    """
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

def predict(text, model, write_to_csv=False, path=None):
    """
    Predict sentiments of sentences in a given text. The function first tokenizes sentences, make predictions and write
    results.
    Parameters
    ----------
    text: string
        text to be analyzed
    model: BertForSequenceClassification
        path to the classifier model
    write_to_csv (optional): bool
    path (optional): string
        path to write the string
    """
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert") #get the tokenizer from the prosusAI repo
    sentences = sent_tokenize(text)

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score']) #initiate a panda df
    for batch in chunks(sentences, 5): #5 = length of sequence. it has no influence on the result
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        features = convert_examples_to_features(examples, label_list, 64, tokenizer)
    
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        with torch.no_grad():
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0] #calculate the tensors using the finbert model from the prosusAI repo
            logging.info(logits)
            logits = softmax(np.array(logits)) #normalize logits to the intervall [0,1] --> thus the sentiment score is bounded between [-1,1]
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1]) #positive tensor - negative sensor = Sentiment score [-1,1]
            predictions = np.squeeze(np.argmax(logits, axis=1)) #taking the maximum of the calculated tensors to predict wether the input is positive, negative or neutral

            batch_result = {'sentence': batch,
                            'logit': list(logits),
                            'prediction': predictions,
                            'sentiment_score': sentiment_score} #define the columns of the dataframe

            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True) #hand over the results to the dataframe

    result['prediction'] = result.prediction.apply(lambda x: label_dict[x])
    if write_to_csv:
        result.to_csv(path, sep=',', index=False)
    return result


df=pd.read_csv("df.csv", sep=",")

df["Headline"] = df["Headline"].map('{}.'.format)
df["Headline"] = df["Headline"].convert_dtypes(convert_string=True)

df_sent=df["Headline"].to_frame()
df_sent = " ".join(df_sent["Headline"])
#print(test)
print(predict(text=df_sent,
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert"), write_to_csv = True, path="./Full_Data_test.csv"))