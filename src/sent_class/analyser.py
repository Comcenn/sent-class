from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self
from transformers import Pipeline, pipeline


@dataclass
class SentimentAnalysisResult:
    label: str
    score: float


class SentimentAnalyser(ABC):
    
    @classmethod
    @abstractmethod
    def build(cls) -> Self:
        ...
    
    @abstractmethod
    def analyse_string(self, text: str) -> SentimentAnalysisResult:
        ...


class HuggingFaceSentimentAnalyser(SentimentAnalyser):
    classifier: Pipeline

    def __init__(self, classifier: Pipeline):
        super().__init__()
        self.classifier = classifier
    
    @classmethod
    def build(cls) -> Self:
        classifier = pipeline("sentiment-analysis")
        return cls(classifier)

    def analyse_string(self, text: str) -> SentimentAnalysisResult:
        result = self.classifier(text)
        print(result)
        return SentimentAnalysisResult(label=result[0]['label'], score=result[0]['score'])