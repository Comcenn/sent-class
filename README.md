# sent-class
Python Sentiment Analyser
## Features

- Perform sentiment analysis using the Hugging Face Transformers library.
- Includes a `HuggingFaceSentimentAnalyser` class for analyzing text sentiment.
- Provides test cases to ensure the reliability of the sentiment analysis.

## Installation

This project uses the [UV package manager](https://docs.astral.sh/uv/). To install the dependencies, run in the root directory:

```bash
uv sync
```

## Usage

### Sentiment Analysis

The `HuggingFaceSentimentAnalyser` class can be used to analyze the sentiment of a given text. Here's an example:

```python
from sent_class.analyser import HuggingFaceSentimentAnalyser

analyser = HuggingFaceSentimentAnalyser.build()
result = analyser.analyse_string("I love programming!")
print(f"Label: {result.label}, Score: {result.score}")
```

### Running Tests

To run the test suite, use:

```bash
pytest
```

## Requirements

- Python 3.13 or higher
- Dependencies listed in `pyproject.toml`

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.