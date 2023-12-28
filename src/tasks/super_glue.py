"""
https://huggingface.co/datasets/super_glue
"""
import json
from typing import Any, Dict

from .datasets_preprocess import DatasetPreprocessor, preprocess


class boolq(DatasetPreprocessor):
    def preprocess(self, passage: str, question: str, label: int):
        assert isinstance(passage, str)
        assert isinstance(question, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            passage=passage, question=question
        )
        if label in [0, 1]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["passage"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["passage"], example["question"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for passage, question, label in zip(
                example["passage"], example["question"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(passage, question, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class cb(DatasetPreprocessor):
    def preprocess(self, premise: str, hypothesis: str, label: int):
        assert isinstance(premise, str)
        assert isinstance(hypothesis, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            premise=premise, hypothesis=hypothesis
        )
        if label in [0, 1, 2]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["premise"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["premise"], example["hypothesis"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for premise, hypothesis, label in zip(
                example["premise"], example["hypothesis"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(premise, hypothesis, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class copa(DatasetPreprocessor):
    def preprocess(self, premise: str, choice1: str, choice2, label: int, **kwargs):
        assert isinstance(premise, str)
        assert isinstance(choice1, str)
        assert isinstance(choice2, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            premise=premise, choice1=choice1, choice2=choice2
        )
        if label in [0, 1]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["premise"], str):
            # not batched
            input_text, target_text = self.preprocess(**example)
        else:
            # batched
            input_text, target_text = [], []
            for sample in zip(
                example["premise"],
                example["choice1"],
                example["choice2"],
                example["label"],
            ):
                _input_text, _target_text = self.preprocess(*sample)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class multirc(DatasetPreprocessor):
    def preprocess(
        self,
        paragraph: str,
        answer: str,
        question: str,
        label: int,
        **kwargs,
    ):
        assert isinstance(paragraph, str)
        assert isinstance(answer, str)
        assert isinstance(question, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            paragraph=paragraph,
            answer=answer,
            question=question,
        )
        if label in [0, 1]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["paragraph"], str):
            # not batched
            input_text, target_text = self.preprocess(**example)
        else:
            # batched
            input_text, target_text = [], []
            for sample in zip(
                example["paragraph"],
                example["answer"],
                example["question"],
                example["label"],
            ):
                _input_text, _target_text = self.preprocess(*sample)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class rte(DatasetPreprocessor):
    def preprocess(self, premise: str, hypothesis: str, label: int):
        assert isinstance(premise, str)
        assert isinstance(hypothesis, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            premise=premise, hypothesis=hypothesis
        )
        if label in [0, 1]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["premise"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["premise"], example["hypothesis"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for premise, hypothesis, label in zip(
                example["premise"], example["hypothesis"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(premise, hypothesis, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class wic(DatasetPreprocessor):
    def preprocess(self, word: str, sentence1: str, sentence2: str, label: int):
        assert isinstance(word, str)
        assert isinstance(sentence1, str)
        assert isinstance(sentence2, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            word=word, sentence1=sentence1, sentence2=sentence2
        )
        if label in [0, 1]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["sentence1"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["word"],
                example["sentence1"],
                example["sentence2"],
                example["label"],
            )
        else:
            # batched
            input_text, target_text = [], []
            for word, sentence1, sentence2, label in zip(
                example["word"],
                example["sentence1"],
                example["sentence2"],
                example["label"],
            ):
                _input_text, _target_text = self.preprocess(
                    word, sentence1, sentence2, label
                )
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class wsc(DatasetPreprocessor):
    def preprocess(self, text: str, span1_text: str, span2_text: str, label: int):
        assert isinstance(text, str)
        assert isinstance(span1_text, str)
        assert isinstance(span2_text, str)
        assert isinstance(label, int)
        input_text = self.template["input_text"].format(
            text=text, span1_text=span1_text, span2_text=span2_text
        )
        if label in [0, 1]:
            target_text = self.template["target_text"][str(label)]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["text"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["text"],
                example["span1_text"],
                example["span2_text"],
                example["label"],
            )
        else:
            # batched
            input_text, target_text = [], []
            for text, span1, span2, label in zip(
                example["text"],
                example["span1_text"],
                example["span2_text"],
                example["label"],
            ):
                _input_text, _target_text = self.preprocess(text, span1, span2, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )
