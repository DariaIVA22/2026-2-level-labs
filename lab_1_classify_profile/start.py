"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
from lab_1_classify_profile.main import (tokenize, remove_stop_words, calculate_frequencies, get_top_n_words)

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

    result = top_7
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
