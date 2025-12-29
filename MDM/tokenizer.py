import json


class SimpleTokenizer:
    """
    A simple vocabulary-based tokenizer.
    Assumes:
      - The vocabulary contains full tokens (words, special tokens…)
      - No subword or BPE logic
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        # Define optional helpers
        self.pad_token = "<PAD>" if "<PAD>" in vocab else None
        self.start_token = "<START>" if "<START>" in vocab else None
        self.end_token = "<END>" if "<END>" in vocab else None
        self.mask_token = "<MASK>" if "<MASK>" in vocab else None
        self.vocab_size = len(vocab)

    @staticmethod
    def from_file(path):
        """Load vocabulary from tokenizer.json"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SimpleTokenizer(data["vocab"])

    def encode(self, text, add_special_tokens=False):
        """
        Encoding with punctuation handling.
        - Handles punctuation attached to words (e.g., "terpus." -> "terpus" + ".")
        """
        ids = []
        if add_special_tokens and self.start_token:
            ids.append(self.token2id[self.start_token])

        words = text.split(" ")
        
        for word in words:
            # Strip whitespace (newlines, tabs, etc.)
            word = word.strip()
            
            # Skip empty words
            if not word:
                continue
            
            # Handle punctuation: check if word ends with punctuation
            if word[-1] in '.?':
                base_word = word[:-1]
                punct = word[-1]
                
                # Add base word
                if base_word in self.token2id:
                    ids.append(self.token2id[base_word])
                else:
                    raise ValueError(f"Unknown token: '{base_word}'")
                
                # Add punctuation
                if punct in self.token2id:
                    ids.append(self.token2id[punct])
                else:
                    raise ValueError(f"Unknown punctuation: '{punct}'")
            else:
                # No punctuation, add word as-is
                if word in self.token2id:
                    ids.append(self.token2id[word])
                else:
                    raise ValueError(f"Unknown token: '{word}'")

        if add_special_tokens and self.end_token:
            ids.append(self.token2id[self.end_token])

        return ids

    def decode(self, ids, skip_special_tokens=False):
        """
        Convert token IDs back to text, removing spaces before punctuation
        """
        toks = []
        for i in ids:
            tok = self.id2token[i]
            if skip_special_tokens and tok.startswith("<") and tok.endswith(">"):
                continue
            toks.append(tok)

        # Join tokens and remove spaces before punctuation
        result = " ".join(toks)
        result = result.replace(" ?", "?")
        result = result.replace(" .", ".")
        return result

    def __len__(self):
        return len(self.vocab)


# # Small test when running tokenizer.py directly
# if __name__ == "__main__":
#     tokenizer = SimpleTokenizer.from_file("tokenizer.json")

#     example = "Alex is a bumpus"
#     try:
#         print("Encoding:", tokenizer.encode(example))
#     except Exception as e:
#         print("Error:", e)
