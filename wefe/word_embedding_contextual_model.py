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
    print(res)
    res = self.tokenizer(word,return_tensors="pt")
    start_idx = 0 if self.use_cls else 1
    res.values.map(lambda x: x[None,0,start_idx:-1)
    print(res)
    
    # res["input_ids"] = res["input_ids"][None,0,start_idx:-1]
    # res["token_type_ids"] = res["token_type_ids"][None,0,start_idx:-1]
    # res["attention_mask"] = res["attention_mask"][None,0,start_idx:-1]
    return res
    
    

  def __getitem__(self, word):
    word_tokens = self.get_word_tokens(word)
    return self.aggregation_method(
      self.wv(**word_tokens)["last_hidden_state"].detach().numpy(),
      axis=1
    )[0]