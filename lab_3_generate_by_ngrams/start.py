"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable
from lab_3_generate_by_ngrams.main import TextProcessor, NGramLanguageModel


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    t = TextProcessor("_")
    encoded_text = t.encode(text)
    decoded_text = t.decode(encoded_text)
    n = NGramLanguageModel(encoded_text, 3)
    build_example =n.build()
    print(f"build example : {build_example}")
    freq_example = n._n_gram_frequencies
    print(f"frequencies example: {freq_example}")
    result = decoded_text
    assert result



if __name__ == "__main__":
    main()
