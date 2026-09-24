"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
from lab_1_classify_profile.main import (tokenize, remove_stop_words, calculate_frequencies, get_top_n_words, create_language_profile, check_profile, compare_profiles_by_top_n, detect_language_by_top_n)

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
    print("Токены:", tokens)

    clean_tokens = remove_stop_words(tokens, stopwords)
    print("Токены без стоп-слов:", clean_tokens)

    freq_dict = calculate_frequencies(clean_tokens)
    print("Частотный словарь:", freq_dict)

    top_7 = get_top_n_words(freq_dict, 7)
    print("Топ-7 популярных слов текста:", top_7)

    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    unknown_profile = create_language_profile("unknown", unknown_text, stopwords)

    print("de_profile валиден:", check_profile(de_profile))
    print("en_profile валиден:", check_profile(en_profile))
    print("unknown_profile валиден:", check_profile(unknown_profile))

    if de_profile is None or en_profile is None or unknown_profile is None:
        print("Ошибка")
        return

    detected_language = detect_language_by_top_n(
        unknown_profile, de_profile, en_profile, 15
    )
    print("Язык:", detected_language)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
