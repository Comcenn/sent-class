"""Sentiment analysis module"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Self

from transformers import Pipeline, pipeline


LOGGER = getLogger(__name__)


class Labels(StrEnum):
    """Sentiment labels used in the analysis."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


LABEL_MULTIPIERS = {
    Labels.POSITIVE: 1,
    Labels.NEGATIVE: -1,
    Labels.NEUTRAL: 0,
}


@dataclass
class SentimentAnalysisResult:
    """Result of sentiment analysis.
    Attributes:
        label: The sentiment label (e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL').
        score: A value between -1 and 1, with -1 being very negative and 1 being very positive.
        Based on the classifiers and their confidences

    """

    label: str
    score: float

    @classmethod
    def _calculate_continuous_score(cls, classification_list) -> float:
        """A simple calc to convert discrete labels to a continuous score. between -1 and 1."""
        return sum(
            prob["score"] * LABEL_MULTIPIERS[prob["label"]]
            for prob in classification_list
        )

    @classmethod
    def from_classifier(cls, classification_list: list[dict]) -> Self:
        """Create a SentimentAnalysisResult from a list of classification results."""
        label = max(classification_list, key=lambda x: x["score"])["label"]
        score = cls._calculate_continuous_score(classification_list)
        return cls(label=label, score=score)


class SentimentAnalyser(ABC):
    """Abstract base class for sentiment analysers."""

    @classmethod
    @abstractmethod
    def build(cls, opts: dict | None = None) -> Self:
        """Factory method to create an instance of the analyser."""

    @abstractmethod
    def analyse_string(self, text: str) -> SentimentAnalysisResult:
        """Analyse the sentiment of a given string."""


class TwitterRobertaBaseSentimentAnalyser(SentimentAnalyser):
    """Sentiment analyser using the 'cardiffnlp/twitter-roberta-base-sentiment'
    model from Hugging Face. This model is specifically fine-tuned for sentiment
    analysis on Twitter data.
    Model Link: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

    How to use:
    ```
        analyser = TwitterRobertaBaseSentimentAnalyser.build()
        result = analyser.analyse_string("I love programming! 🤗")
    ```
    """

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    classifier: Pipeline
    label_mapping = {
        "LABEL_0": Labels.NEGATIVE,
        "LABEL_1": Labels.NEUTRAL,
        "LABEL_2": Labels.POSITIVE,
    }

    def __init__(self, classifier: Pipeline):
        super().__init__()
        self.classifier = classifier

    @classmethod
    def build(cls, opts: dict | None = None) -> Self:
        LOGGER.debug("Building %s", cls.__name__)
        classifier = pipeline(
            task="text-classification",
            model=cls.model_name,
            return_all_scores=True,
            **opts or {},
        )
        return cls(classifier)

    def analyse_string(self, text: str) -> SentimentAnalysisResult:
        """Analyses the sentiment of the input string.
        Args:
            text: The input string to be analysed.
        Returns:
            SentimentAnalysisResult: The result of the sentiment analysis.
        Raises:
            ValueError: If the classifier does not return a list of classification results.
        """
        LOGGER.debug("Analysing text: %s", text)
        result = self.classifier(text)
        LOGGER.debug("Analysis completed, output: %s", result)
        if not isinstance(result, list):
            raise ValueError(
                f"Expected a list of classification results, got {type(result)}."
            )
        result = [
            {"label": self.label_mapping[prob["label"]], "score": prob["score"]}
            for prob in result[0]
        ]
        LOGGER.debug("Mapped labels: %s", result)
        return SentimentAnalysisResult.from_classifier(result)
