"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable
from lab_2_tokenize_by_bpe.main import (
    collect_frequencies,
    decode,
    encode,
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

    word_frequencies = collect_frequencies(text, None, "</s>")
    if word_frequencies is None:
        return
    print(f'Unique words before train: {len(word_frequencies)}')

    trained = train(word_frequencies, 100)
    if trained is None:
        return
    print(f"Unique words after train: {len(trained)}")

    vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")
    if vocabulary is None:
        return
    print(f"Vocabulary size: {len(vocabulary)}")

    with open("lab_2_tokenize_by_bpe/assets/secrets/secret_1.txt", "r", encoding="utf-8") as f:
        secret_encoded = [int(x) for x in f.read().split()]
        decoded = decode(secret_encoded, vocabulary, "</s>")
    if decoded is None:
        return
    print("Decoded secret:", decoded)

    with open("lab_2_tokenize_by_bpe/assets/ru_raw.txt", "r", encoding="utf-8") as text_file:
        ru_text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/ru_encoded.txt", "r", encoding="utf-8") as text_file:
        ru_encoded_raw = text_file.read()

    encoded = encode(
        ru_text,
        vocabulary,
        start_of_word_token="\u2581",
        end_of_word_token=None,
        unknown_token="<unk>",
    )
    if encoded is None:
        return

    reference = [int(x) for x in ru_encoded_raw.split()]

    print("First 20 encoded:  ", encoded[:20])
    print("First 20 reference:", reference[:20])
    print("First 20 match:", encoded[:20] == reference[:20])
    print("Full match:", encoded == reference)


if __name__ == "__main__":
    main()
