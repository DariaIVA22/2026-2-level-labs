"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable
from lab_2_tokenize_by_bpe.main import (
    collect_frequencies,
    decode,
    encode,
    get_vocabulary,
    load_vocabulary,
    train,
)


def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8") as text_file:
        text_reference_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8") as text_file:
        translation_encoded_raw = text_file.read()
    with open(
        "lab_2_tokenize_by_bpe/assets/secrets/secret_1.txt",
        "r",
        encoding="utf-8",
    ) as secret_file:
        secret_text = [int(x) for x in secret_file.read().split()]
    with open("lab_2_tokenize_by_bpe/assets/ru_raw.txt", "r", encoding="utf-8") as text_file:
        ru_text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/ru_encoded.txt", "r", encoding="utf-8") as text_file:
        ru_expected = [int(x) for x in text_file.read().split()]

    word_frequencies = collect_frequencies(text, None, "</s>")
    assert word_frequencies, "collect_frequencies failed"
    print(f"Frequency of words before train: {len(word_frequencies)}")

    trained = train(word_frequencies, 100)
    assert trained, "train failed"
    print(f"Unique words after train: {len(trained)}")

    trained_vocab = get_vocabulary(trained, "<unk>")
    assert trained_vocab, "get_vocabulary failed"
    print(f"Trained vocabulary size: {len(trained_vocab)}")

    decoded = decode(secret_text, trained_vocab, "</s>")
    assert decoded, "decode failed"
    print(f"Decoded secret: {decoded}")

    vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")
    assert vocabulary, "load_vocabulary failed"
    print(f"Vocabulary size: {len(vocabulary)}")

    encoded = encode(
        ru_text,
        vocabulary,
        start_of_word_token="▁",
        end_of_word_token=None,
        unknown_token="<unk>",
    )
    assert encoded, "encode failed"

    print("First 20 encoded:  ", encoded[:20])
    print("First 20 reference:", ru_expected[:20])
    print("First 20 match:", encoded[:20] == ru_expected[:20])
    print("Full match:", encoded == ru_expected)

    result = encoded
    assert result, "Translation not working"


if __name__ == "__main__":
    main()