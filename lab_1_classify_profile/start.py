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

    tokens = tokenize(de_text)
    assert tokens is not None
    print("Tokens:", tokens)

    clean_tokens = remove_stop_words(tokens, stopwords)
    assert clean_tokens is not None
    print("Tokens without stop-words:", clean_tokens)

    freq_dict = calculate_frequencies(clean_tokens)
    assert freq_dict is not None
    print("Frequency dictionary:", freq_dict)

    top_7 = get_top_n_words(freq_dict, 7)
    assert freq_dict is not None
    print("Top-7 words:", top_7)

    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    unknown_profile = create_language_profile("unknown", unknown_text, stopwords)
    assert de_profile is not None
    assert en_profile is not None
    assert unknown_profile is not None

    detected_language = detect_language_by_top_n(
        unknown_profile, de_profile, en_profile, 15
    )
    assert detected_language is not None
    print("Language:", detected_language)

    detected_language_mse = detect_language_by_mse(
        unknown_profile, de_profile, en_profile
    )
    assert detected_language_mse is not None
    print("Language by MSE:", detected_language_mse)


if __name__ == "__main__":
    main()
