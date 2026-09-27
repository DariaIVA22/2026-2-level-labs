"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code, too-many-return-statements
from lab_1_classify_profile.main import (
    calculate_frequencies,
    check_profile,
    create_language_profile,
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
    if not isinstance(tokens, (list, tuple)):
        return None
    print("Tokens:", tokens)


    clean_tokens = remove_stop_words(tokens, stopwords)
    if not isinstance(clean_tokens, (list, tuple)):
        return None
    print("Tokens without stop-words:", clean_tokens)

    freq_dict = calculate_frequencies(clean_tokens)
    if not isinstance(freq_dict, dict):
        return None
    print("Frequency dictionary:", freq_dict)

    result = get_top_n_words(freq_dict, 7)
    if not isinstance(result, (list, tuple)):
        return None
    print("Top-7 words:", result)

    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    unknown_profile = create_language_profile("unknown", unknown_text, stopwords)

    if not (isinstance(de_profile, tuple)
            and isinstance(en_profile, tuple)
            and isinstance(unknown_profile, tuple)):
        return None
    print("de_profile is valid:", check_profile(de_profile))
    print("en_profile is valid:", check_profile(en_profile))
    print("unknown_profile is valid:", check_profile(unknown_profile))

    detected_language = detect_language_by_top_n(
        unknown_profile, de_profile, en_profile, 15
    )
    if not isinstance(detected_language, str):
        return None
    print("Language:", detected_language)

    assert result, "Detection result is None"
    return None


if __name__ == "__main__":
    main()
