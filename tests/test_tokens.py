"""The byte-level tokenizer: any text in, the same text out."""

import unittest

import torch

from nanorl import tokens


class RoundTripTests(unittest.TestCase):
    def test_plain_english_survives(self):
        seq = tokens.encode("hello world")
        self.assertEqual(tokens.decode(seq), "hello world")

    def test_emoji_and_hindi_survive(self):
        text = "नमस्ते दुनिया 🌍 ok"
        self.assertEqual(tokens.decode(tokens.encode(text)), text)

    def test_an_empty_string_still_has_its_bookends(self):
        seq = tokens.encode("")
        self.assertEqual(seq, [tokens.BOS, tokens.EOS])
        self.assertEqual(tokens.decode(seq), "")

    def test_every_id_is_inside_the_vocabulary(self):
        seq = tokens.encode("anything at all — 123 — 日本語")
        self.assertTrue(all(0 <= i < tokens.VOCAB_SIZE for i in seq))

    def test_the_specials_frame_the_sequence(self):
        seq = tokens.encode("hi")
        self.assertEqual(seq[0], tokens.BOS)
        self.assertEqual(seq[-1], tokens.EOS)

    def test_specials_are_never_decoded_as_text(self):
        seq = tokens.encode("mixed") + [tokens.PAD, tokens.PAD]
        self.assertEqual(tokens.decode(seq), "mixed")


class PadBatchTests(unittest.TestCase):
    def test_shapes_and_masks(self):
        batch, mask = tokens.pad_batch(["hi", "longer sentence here"], 32)
        self.assertEqual(batch.shape, (2, 32))
        self.assertEqual(mask.shape, (2, 32))
        # "hi" is BOS + h + i + EOS = 4 real tokens.
        self.assertEqual(int(mask[0].sum()), 4)
        self.assertEqual(int(mask[1].sum()), len("longer sentence here") + 2)

    def test_padding_uses_the_pad_id_exactly_where_the_mask_is_zero(self):
        batch, mask = tokens.pad_batch(["abc"], 10)
        real = int(mask[0].sum())
        self.assertEqual(int(batch[0][real:].sum()), tokens.PAD * (10 - real))

    def test_a_long_text_is_cut_and_still_ends_with_eos(self):
        batch, mask = tokens.pad_batch(["x" * 100], 20)
        self.assertEqual(int(mask[0].sum()), 20)
        self.assertEqual(int(batch[0][-1]), tokens.EOS)

    def test_decoding_a_padded_batch_loses_nothing(self):
        text = "take me back"
        batch, mask = tokens.pad_batch([text], 64)
        self.assertEqual(tokens.decode(batch[0].tolist()), text)


if __name__ == "__main__":
    unittest.main()
