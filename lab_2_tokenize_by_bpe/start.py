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
    result = None

    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    with open("lab_2_tokenize_by_bpe/assets/ru_raw.txt", "r", encoding="utf-8") as text_file:
        ru_text = text_file.read()

    with open("lab_2_tokenize_by_bpe/assets/ru_encoded.txt", "r", encoding="utf-8") as text_file:
        ru_expected = [int(value) for value in text_file.read().split()]

    secret_path = "lab_2_tokenize_by_bpe/assets/secrets/secret_1.txt"
    with open(secret_path, "r", encoding="utf-8") as secret_file:
        secret_raw = secret_file.read()

    secret_text = []
    for value in secret_raw.split():
        if value.lstrip("-").isdigit():
            secret_text.append(int(value))

    word_frequencies = collect_frequencies(text, start_of_word=None, end_of_word="</s>")
    assert word_frequencies, "collect_frequencies failed"

    trained = train(word_frequencies, num_merges=100)
    assert trained, "train failed"

    trained_vocab = get_vocabulary(trained, unknown_token="<unk>")
    assert trained_vocab, "get_vocabulary failed"

    decoded = decode(secret_text, trained_vocab, "</s>")
    assert decoded, "decode failed"
    print("Decoded secret:", decoded)

    vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")
    assert vocabulary, "load_vocabulary failed"

    encoded = encode(ru_text, vocabulary, "▁", None, "<unk>")
    assert encoded, "encode failed"

    compare_len = min(20, len(encoded), len(ru_expected))
    is_match = all(encoded[i] == ru_expected[i] for i in range(compare_len))
    print("Result of matching:", is_match)

    result = encoded
    assert result, "Translation not working"


if __name__ == "__main__":
    main()
