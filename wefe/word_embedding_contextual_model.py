import numpy as np
from wefe.word_embedding_base_model import WordEmbeddingBaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class WordEmbeddingContextualModel(WordEmbeddingBaseModel):
  aggregation_methods = {
      "mean": np.mean,
      "sum": np.sum
  }
  def __init__(
    self, tokenizer, hugging_face_pretrained_model, use_cls=True, name = None, aggregation="mean"):
    if isinstance(hugging_face_pretrained_model,PreTrainedModel): 
      self.wv = hugging_face_pretrained_model
    if isinstance(tokenizer,PreTrainedTokenizerBase): 
      self.tokenizer = tokenizer
    
    self.use_cls = use_cls
    self.name = name
    self.aggregation_method = self.aggregation_methods[aggregation]
    self.vocab_prefix = None
  
  def get_word_tokens(self, word):
    res = self.tokenizer(word,return_tensors="pt")
    
     
    #  Removing CLS depending on parameter
    start_idx = 0 if self.use_cls else 1

    # Ignoring all the other vectors when use_cls is on
    end_idx = 1 if self.use_cls else -1

    res["input_ids"] = res["input_ids"][None,0,start_idx:end_idx]
    res["token_type_ids"] = res["token_type_ids"][None,0,start_idx:end_idx]
    res["attention_mask"] = res["attention_mask"][None,0,start_idx:end_idx]
    return res
    
    

  def __getitem__(self, word):
    word_tokens = self.get_word_tokens(word)
    
    res = self.wv(**word_tokens)["last_hidden_state"].detach().numpy()[0]
    
    # If use_cls is off then we aggregate
    if not self.use_cls:
      return self.aggregation_method(res,axis=0)

    return res[0]