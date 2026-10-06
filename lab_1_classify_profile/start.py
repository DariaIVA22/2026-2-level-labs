"""
Language detection starter.
"""
from lab_1_classify_profile.main import (
    calculate_frequencies,
    create_language_profile,
    detect_language_by_mse,
    detect_language_by_top_n,
    get_top_n_words,
    remove_stop_words,
    tokenize,
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()

    result = None

    tokens = tokenize(de_text)
    assert tokens is not None
    print("Tokens:", tokens)
    assert freq_dict is not None
    print("Frequency dictionary:", freq_dict)

    top_7 = get_top_n_words(freq_dict, 7)
    assert top_7 is not None
    print("Top-7 words:", top_7)
    result = get_top_n_words(freq_dict, 7)
    assert result is not None
    print("Top-7 words:", result)

    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    assert detected_language_mse is not None
    print("Language by MSE:", detected_language_mse)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()