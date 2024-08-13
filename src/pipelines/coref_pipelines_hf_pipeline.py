from transformers import Pipeline
from torch.nn.functional import softmax
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AlbertForSequenceClassification


class DiscoursePipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        for arg in []:
            if arg in kwargs.keys():
                preprocess_kwargs[arg] = kwargs[arg]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text):
        return self.tokenizer(text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        probabilities = softmax(model_outputs.logits, dim=1)
        best_classes = probabilities.argmax(dim=1).numpy()
        labels = list(map(self.model.config.id2label.get, best_classes))
        scores = probabilities[:, best_classes].numpy()
        return {"labels": labels[0], "scores": float(scores[0])}


PIPELINE_REGISTRY.register_pipeline(
    "discourse-classification",
    pipeline_class=DiscoursePipeline,
    pt_model=AlbertForSequenceClassification,
)