#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

# defaultdict
from collections import defaultdict 

# string
import string

# random
import random

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

# PART 1 (finished)
def get_candidates(lemma, pos) -> List[str]:
    synonym_list = []

    for item in wn.lemmas(lemma, pos):
      s = item.synset()
      for word in s.lemmas():
        # Check if it's the original word
        if word.name() != lemma:
          # Replace all underlines with blankspaces
          synonym_list.append(word.name().replace('_',' '))

    # Turn the list into a set to eliminate duplicated items
    synonym = set(synonym_list)
    return synonym 

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

# PART 2 (finished)
def wn_frequency_predictor(context : Context) -> str:
    synonym_dict = defaultdict(int)

    for lemma in wn.lemmas(context.lemma, context.pos):
        s = lemma.synset()
        for word in s.lemmas():
          # Check if it's the original word
          if word.name() != context.lemma:
            # Replace all underlines with blankspaces and take counts
            new_name = word.name().replace('_',' ')
            synonym_dict[new_name] += word.count()

    return max(synonym_dict, key= lambda x: synonym_dict[x]) 

# PART 3 (finished)
def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = stopwords.words('english')
    punctuation = string.punctuation

    max_overlap = 0
    best_candidate = ""
    word_context = context.left_context + context.right_context

    for lemma in wn.lemmas(context.lemma, context.pos):
      synset = lemma.synset()
      synonym_name = synset.lemmas()[0].name()
      overlap_length = 0

      if synonym_name != context.lemma:
        definition_list = [synset.definition()]
        # len(definition_list) == 1
        definition_list += synset.examples()
        # len(example_list) == 0, 1, or more than 1

        # len(synset.hypernyms()) == 0, 1, or more that 1
        for item in synset.hypernyms():
          definition_list.append(item.definition())
          for example in item.examples():
            definition_list.append(example)

        # compare the overlapping items
        # need to tokenize
        for definition in definition_list:
          overlap = set(tokenize(definition)).intersection(set(word_context)) - set(stop_words) - set(punctuation)
          overlap_length += len(overlap)

        if overlap_length > max_overlap:
          max_overlap = len(overlap)
          best_candidate = synonym_name.replace('_',' ')        

    # if still no word
    if max_overlap == 0:

      synonym_dict = defaultdict(int)

      for lemma in wn.lemmas(context.lemma, context.pos):
          s = lemma.synset()
          for word in s.lemmas():
            # Check if it's the original word
            if word.name() != context.lemma:
              # Replace all underlines with blankspaces and take counts
              new_name = word.name().replace('_',' ')
              synonym_dict[new_name] += word.count()

      best_candidate = max(synonym_dict, key= lambda x: synonym_dict[x]) 

    return best_candidate       
   
# PART 4 (finished)
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        synonym = get_candidates(context.lemma, context.pos)
        max_similarity = 0.0
        best_candidate = ""

        for word in synonym:
          try:
            if self.model.similarity(context.lemma, word) > max_similarity:
              max_similarity = self.model.similarity(context.lemma, word)
              best_candidate = word
          except:
            continue

        return best_candidate

# PART 5 (finished)
class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        synonym = get_candidates(context.lemma, context.pos)
        masked_id = len(context.left_context) + 1

        sentence = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        input_toks = self.tokenizer.convert_tokens_to_ids(sentence)

        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][masked_id])[::-1]

        for word in self.tokenizer.convert_ids_to_tokens(best_words):
          if word in synonym:
            return word

# PART 6 
class Part6Predictor(object):

  # This predictor is an advanced version of the one used in Part 5:
  # 
  # It produces predictions in two ways:
  # (1) with [MASK];
  # (2) without [MASK] (with the original lemma word).
  # 
  # Then, the two predictions are merged and sorted
  # to choose the best prediction.

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        synonym = get_candidates(context.lemma, context.pos)
        masked_id = len(context.left_context) + 1

        sentence_mask = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        sentence_original = ['[CLS]'] + context.left_context + [context.lemma] + context.right_context + ['[SEP]']
        input_toks_mask = self.tokenizer.convert_tokens_to_ids(sentence_mask)
        input_toks_original = self.tokenizer.convert_tokens_to_ids(sentence_original)

        input_mat_mask = np.array(input_toks_mask).reshape((1,-1))
        input_mat_original = np.array(input_toks_original).reshape((1,-1))
        outputs_mask = self.model.predict(input_mat_mask)
        outputs_original = self.model.predict(input_mat_original)

        predictions_mask = outputs_mask[0]
        predictions_original = outputs_original[0]
        predictions = predictions_mask + predictions_original
        best_words = np.argsort(predictions[0][masked_id])[::-1]

        for word in self.tokenizer.convert_ids_to_tokens(best_words):
          if word in synonym:
            return word

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # PART 1 test
    # =============
    """
    print(get_candidates('slow','a'))
    compare_set = {'deadening', 'tiresome', 'sluggish', 'dense', 'tedious', 'irksome', 'boring', 'wearisome', 'obtuse', 'dim', 'dumb', 'dull', 'ho-hum'}
    print(get_candidates('slow','a').difference(compare_set))
    """

    # PART 2 test
    # =============
    """
    for context in read_lexsub_xml(sys.argv[1]):
        prediction = wn_frequency_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    """

    """
    ---------------------------------------
    Total = 298, attempted = 298
    precision = 0.098, recall = 0.098
    Total with mode 206 attempted 206
    precision = 0.136, recall = 0.136
    ---------------------------------------
    """

    # PART 3 test
    # =============
    """
    for context in read_lexsub_xml(sys.argv[1]):
        prediction = wn_simple_lesk_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    """

    """
    ---------------------------------------
    Total = 298, attempted = 298
    precision = 0.095, recall = 0.095
    Total with mode 206 attempted 206
    precision = 0.136, recall = 0.136
    ---------------------------------------
    """

    # PART 4 test
    # =============
    """
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict_nearest(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    """

    """
    ---------------------------------------
    Total = 298, attempted = 298
    precision = 0.115, recall = 0.115
    Total with mode 206 attempted 206
    precision = 0.170, recall = 0.170
    ---------------------------------------
    """

    # PART 5 test
    # =============
    """
    predictor = BertPredictor()
    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    """

    """
    ---------------------------------------
    Total = 298, attempted = 298
    precision = 0.123, recall = 0.123
    Total with mode 206 attempted 206
    precision = 0.184, recall = 0.184
    ---------------------------------------
    """

    # ====FINAL====
    # PART 6 test
    # =============

    # This predictor is an advanced version of the one used in Part 5:
    # 
    # It produces predictions in two ways:
    # (1) with [MASK];
    # (2) without [MASK] (with the original lemma word).
    # 
    # Then, the two predictions are merged and sorted
    # to choose the best prediction.

    #"""
    predictor = Part6Predictor()
    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    #"""

    """
    ---------------------------------------
    Total = 298, attempted = 298
    precision = 0.144, recall = 0.144
    Total with mode 206 attempted 206
    precision = 0.204, recall = 0.204
    ---------------------------------------
    """
