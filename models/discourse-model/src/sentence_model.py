import torch.nn as nn
from typing import Optional, List, Dict, Union, Any

from torch.utils.data import Dataset
from transformers import (
    AutoModel, AutoConfig, RobertaConfig, RobertaModel, PreTrainedModel, BertPreTrainedModel,
    PretrainedConfig
)

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from util import _get_attention_mask, label_mapper


class TokenizedDataset(Dataset):
    def __init__(self, doc_list=None, tokenizer=None, do_score=False, max_length=512, label_mapper=label_mapper):
        """
        Processes a dataset, `doc_list` for either training/evaluation or scoring.
        * doc_list: We expect a list of dictionaries.
            * If `do_score=False`, then we are training/evaluating. We need a `label` field:
                [[{'sent': <sent 1>, 'label': <label 1>}, ...]]
            * If `do_score=True`, then we are scoring. We don't need a `label` field:
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
        self.attention = []
        self.categories = []
        self.do_score = do_score  # whether to just score data (i.e. no labels exist)
        self.max_length = max_length
        if not do_score:
            self.process_data(doc_list)
        self.label_mapper = label_mapper
        if self.label_mapper is not None:
            self.idx2label_mapper = {v: k for k, v in self.label_mapper.items()}
            self.num_labels = max(self.label_mapper.values()) + 1

    def process_data(self, doc_list):
        for doc in doc_list:
            input_ids, attention_mask, doc_labels = self.process_one_doc(doc)
            if input_ids is not None:
                self.input_ids.append(input_ids)
                self.attention.append(attention_mask)
                self.labels.append(doc_labels)

    def transform_logits_to_labels(self, logits, num_docs):
        preds = logits.reshape(num_docs, self.num_labels).argmax(dim=1)
        preds = preds.detach().cpu().numpy()
        return list(map(self.idx2label_mapper.get, preds))

    def process_one_doc(self, doc):
        doc_tokens = []
        doc_labels = []
        if self.do_score:
            for sentence in doc['sentences']:
                doc_tokens.append(self.tokenizer.encode(sentence))
        else:
            for sentence, label in zip(doc['sent'], doc['quote_type']):
                doc_tokens.append(self.tokenizer.encode(sentence))
                doc_labels.append(int(self.label_mapper[label]))
        doc_tokens = list(map(torch.tensor, doc_tokens))
        attention_mask = _get_attention_mask(doc_tokens)[:, :self.max_length]
        input_ids = pad_sequence(
            doc_tokens,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )[:, :self.max_length]
        if not self.do_score:
            # we need have the labels like this for this weird `nested_concat` function
            # in the `evaluation_loop` of the `Trainer.py`.
            doc_labels = torch.tensor(doc_labels).unsqueeze(0).to(float)
        else:
            doc_labels = None
        return input_ids, attention_mask, doc_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        output_dict = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention[idx],
        }
        if not self.do_score:
            output_dict['labels'] = self.labels[idx]
        return output_dict


def collate_fn(dataset):
    """
    Takes in an instance of Torch Dataset.
    Returns:
     * input_ids: List len N,  elements are len num toks in doc
     * label_ids: List size N, elements are len num sents in doc
    """
    # transpose dict
    batch_by_columns = {}
    for key in dataset[0].keys():
        batch_by_columns[key] = list(map(lambda d: d[key], dataset))

    return {
        'input_ids': batch_by_columns['input_ids'],
        'attention_mask': batch_by_columns['attention_mask'],
        'labels': batch_by_columns['labels'],
    }


###############################
# model components
class AdditiveSelfAttention(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, input_dim)
        self.ws2 = nn.Linear(input_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ws2.state_dict()['weight'])

    def forward(self, hidden_embeds, context_mask=None):
        ## get sentence encoding using additive attention (appears to be based on Bahdanau 2015) where:
        ##     score(s_t, h_i) = v_a^T tanh(W_a * [s_t; h_i]),
        ## here, s_t, h_i = word embeddings
        ## align(emb) = softmax(score(Bi-LSTM(word_emb)))
        # word_embs: shape = (num sentences in curr batch * max_len * embedding_dim)     # for word-attention:
        #     where embedding_dim = hidden_dim * 2                                       # -------------------------------------
        # sent_embs: shape = if one doc:   (num sentences in curr batch * embedding_dim)
        #         #          if many docs: (num docs x num sentences in batch x max word len x hidden_dim)
        self_attention = torch.tanh(self.ws1(self.drop(hidden_embeds)))         # self attention : if one doc: (num sentences in curr batch x max_len x hidden_dim
                                                                              #   if >1 doc: if many docs: (num docs x num sents x max word len x hidden_dim)
        self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)      # self_attention : (num_sentences in curr batch x max_len)
        if context_mask is not None:
            context_mask = -10000 * (context_mask == 0).float()
            self_attention = self_attention + context_mask                    # self_attention : (num_sentences in curr batch x max_len)
        if len(self_attention.shape) == 1:
            self_attention = self_attention.unsqueeze(0)  # todo: does this cause problems?
        self_attention = self.softmax(self_attention).unsqueeze(1)            # self_attention : (num_sentences in curr batch x 1 x max_len)
        return self_attention


class AttentionCompression(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=hidden_size, dropout=dropout)

    def forward(self, hidden_embs, attention_mask=None):
        ## `'hidden_emds'`: shape = N x hidden_dim
        self_attention = self.self_attention(hidden_embs, attention_mask)  # self_attention = N x 1 x N
        ## batched matrix x batched matrix:
        output_encoding = torch.matmul(self_attention, hidden_embs).squeeze(1)
        return output_encoding


class TransformerContext(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = RobertaConfig(**config)

        self.base_model = AutoModel.from_config(config)
        TransformerContext.base_model_prefix = self.base_model.base_model_prefix
        TransformerContext.config_class = self.base_model.config_class

    def forward(self, cls_embeddings):
        # inputs_embeds: input of shape: (batch_size, sequence_length, hidden_size)
        contextualized_embeds = self.base_model(inputs_embeds=cls_embeddings.unsqueeze(0))[0]
        return contextualized_embeds.squeeze()


class BiLSTMContext(nn.Module):
    def __init__(self, config, num_contextual_layers=2, bidirectional=True):
        super().__init__()
        self.bidirectional = True
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=num_contextual_layers,
            bidirectional=bidirectional
        )
        lstm_output_size = config.hidden_size * 2 if self.bidirectional else config.hidden_size
        self.resize = nn.Linear(lstm_output_size, config.hidden_size)
        # init params
        for name, param in self.lstm.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, cls_embeddings):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cls_embeddings)
        resized = self.resize(lstm_out)
        return resized


def freeze_hf_model(model, freeze_layers):
    def freeze_all_params(subgraph):
        for p in subgraph.parameters():
            p.requires_grad = False

    if isinstance(model, RobertaModel):
        layers = model.encoder.layer
    else:
        layers = model.transformer.h

    if freeze_layers is not None:
        for layer in freeze_layers:
            freeze_all_params(layers[layer])


class SentenceClassificationModel(PreTrainedModel):

    base_model_prefix = ''
    config_class = AutoConfig

    def __init__(self, config, hf_model=None):
        super().__init__(config)

        base_model = AutoModel.from_config(config) if hf_model is None else hf_model
        SentenceClassificationModel.base_model_prefix = base_model.base_model_prefix
        SentenceClassificationModel.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = config.classification_head['num_labels']
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss() if self.num_labels > 1 else BCEWithLogitsLoss()
        self.pooling_method = config.classification_head['pooling_method']
        self.word_attention = AttentionCompression(hidden_size=config.hidden_size, dropout=config.hidden_dropout_prob)
        if getattr(config, 'context_layer', None) == 'transformer':
            self.context_layer = TransformerContext(config.context_config)
        elif getattr(config, 'context_layer', None) == 'bilstm':
            self.context_layer = BiLSTMContext(config)
        self.post_init()

    def post_init(self):
        # during prediction, we don't have to pass this in
        if hasattr(self.config, 'freeze_layers'):
            base_model = getattr(self.base_model_prefix)
            freeze_hf_model(base_model, freeze_layers=self.config.freeze_layers)

    def pool_words(self, hidden, attention_mask):
        if self.pooling_method == 'average':
            return (hidden.T * attention_mask.T).T.mean(axis=1)
        elif self.pooling_method == 'cls':
            return hidden[:, 0, :]
        elif self.pooling_method == 'attention':
            return self.word_attention(hidden, attention_mask)

    def get_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        _, logits = self.process_one_doc(input_ids, attention_mask)
        return logits

    def process_one_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if input_ids is not None and len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        outputs = self.base_model(input_ids, attention_mask=attention_mask)

        # pool word embeddings
        hidden = outputs[0]
        pooled_output = self.pool_words(hidden, attention_mask)
        pooled_output = self.dropout(pooled_output)
        if self.context_layer is not None:
            pooled_output = self.context_layer(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, 1))
            else:
                loss = self.loss_fct(logits, labels.squeeze().to(int))

        # note that this unstacks the logits such that they are a long 1-d array that
        # we have to reshape. This is on purpose, for the HF's EvalPrediction function...
        return loss, logits.view(1, -1)

    def forward(self, input_ids: List[torch.Tensor], attention_mask: List[torch.Tensor],
                labels: Optional[List[torch.Tensor]] = None,):
        outputs = list(map(lambda x: self.process_one_doc(*x), zip(input_ids, attention_mask, labels)))
        losses, logits = list(zip(*outputs))
        loss = None if losses[0] == None else sum(losses)
        logits = torch.vstack(logits)
        return (loss, logits) if loss is not None else logits

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs. Needed for `Trainer.py` class.
        (Custom implementation is necessary because our dataset is in a list format.)
        """
        token_inputs = [tensor for key, tensor in input_dict.items() if "input" in key]
        if token_inputs:
            # torch.vstack is the only change
            return sum([torch.vstack(token_input).numel() for token_input in token_inputs])
        else:
            return 0


if __name__ == "__main__":
    from transformers import AutoModel, AutoConfig, AutoTokenizer

    hf_model = AutoModel.from_pretrained('roberta-base')
    hf_config = AutoConfig.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # roberta_tokenizer.encode('Hello.') + roberta_tokenizer.encode('My name is Alex.')
    # >>> [   0,  31414, 4,  2,  0, 2387, 766, 16, 2618, 4, 2    ]
    #        <s> Hello  . </s> <s>  My   name is  Alex  . </s>
    #        bos  1st sent  eos bos     2nd sent         . eos

    hf_config.num_labels = 1
    sentence_classifier = SentenceClassificationModel(hf_config, hf_model=hf_model)

    import functools

    doc = ['Hello.', 'My name is Alex.']
    sentence_toks = list(map(tokenizer.encode, doc))
    sentence_lens = list(map(len, sentence_toks))
    sentence_toks = functools.reduce(lambda a, b: a + b, sentence_toks)
    sent_tensor = torch.tensor([sentence_toks])
    labels = [0, 1]
    labels_tensor = torch.tensor([labels]).to(float)
    #
    sentence_classifier(input_ids=sent_tensor, sentence_lens=[sentence_lens], labels=labels_tensor)