from pytest import MonkeyPatch, raises

from sent_class.analyser import TwitterRobertaBaseSentimentAnalyser


def test_roberta_sentiment_analyser_with_positive_emoji() -> None:
    analyser = TwitterRobertaBaseSentimentAnalyser.build()
    result = analyser.analyse_string("I love programming! 🤗")
    assert result.label == "POSITIVE"
    assert result.score > 0.9


def test_roberta_sentiment_analyser_with_negative_emoji() -> None:
    analyser = TwitterRobertaBaseSentimentAnalyser.build()
    result = analyser.analyse_string("I hate bugs! 😡")
    assert result.label == "NEGATIVE"
    assert result.score < -0.5


def test_roberta_sentiment_analyser_with_neutral_text() -> None:
    analyser = TwitterRobertaBaseSentimentAnalyser.build()
    result = analyser.analyse_string("Neutral statement.")
    assert result.label in ["NEUTRAL"]
    assert -0.5 < result.score < 0.5


def test_roberta_roberta_analyse_string_raises_value_error(
    monkeypatch: MonkeyPatch,
) -> None:
    analyser = TwitterRobertaBaseSentimentAnalyser.build()
    monkeypatch.setattr(analyser, "classifier", lambda _: "unexpected output")
    with raises(
        ValueError,
        match=f"Expected a list of classification results, got <class 'str'>.",
    ):
        analyser.analyse_string("This will raise an error.")
