from sent_class.analyser import HuggingFaceSentimentAnalyser


def test_hugging_face_sentiment_analyser_with_positive_emoji():
    analyser = HuggingFaceSentimentAnalyser.build()
    result = analyser.analyse_string("I love programming! 🤗")
    assert result.label == "POSITIVE"
    assert result.score > 0.9


def test_hugging_face_sentiment_analyser_with_negative_emoji():
    analyser = HuggingFaceSentimentAnalyser.build()
    result = analyser.analyse_string("I hate bugs! 😡")
    assert result.label == "NEGATIVE"
    assert result.score > 0.9


def test_hugging_face_sentiment_analyser_with_neutral_text():
    analyser = HuggingFaceSentimentAnalyser.build()
    result = analyser.analyse_string("The sky is blue.")
    assert result.label in ["POSITIVE", "NEGATIVE"]  # Depending on the model, it might classify neutral text differently
    assert result.score > 0.5
