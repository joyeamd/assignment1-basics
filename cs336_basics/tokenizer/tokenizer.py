import json
import regex as re
from typing import Iterable, Iterator, Any
import os
import heapq
from functools import lru_cache


class Tokenizer(object):
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None
        """
        self.vocab = vocab
        self.vocab_token_to_id = {v: k for k, v in self.vocab.items()}
        self.merges = merges  # merges in the order they were created
        self.merges_order = {v:i for i, v in enumerate(merges)}
        self.special_tokens = special_tokens
    
    @classmethod    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens. 
        """
        with open(vocab_filepath, 'r', encoding = "utf-8") as f:
            vocab_data = json.read(f)
        vocab_id_to_token = {v: k for k, v in vocab_data.items()}
        merge_tuples = []
        with open(merges_filepath, 'r', encoding = "utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line:
                    parts = line.split(' ')
                    if len(parts) == 2:
                        merge_tuples.append((parts[0], parts[1]))
        return cls.__init__(vocab_id_to_token, merge_tuples, special_tokens)
    
    def split_by_special(self, text, drop_special = True):
        if not self.special_tokens:
            return [text]

        # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        if not drop_special: 
            pattern = f"({pattern})"

        pattern = re.compile(pattern)
        chunks = pattern.split(text)
        return (c for c in chunks if c)  # remove empty strings        
        
    def encode(self, text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        chunk_texts = self.split_by_special(text, False)
        encode_result = list()
        pattern_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern = re.compile(pattern_str)
        for chunk in chunk_texts:
            if self.special_tokens is not None and chunk in self.special_tokens:
                encode_result.append(self.vocab_token_to_id[chunk.encode("utf-8")])
            else:        
                matches = re.finditer(pattern, chunk)
                for match in matches:
                    token = match.group(0)
                    utf8_bytes = list(bytes([i]) for i in token.encode('utf-8'))
                    while True:
                        #have_pairs = False
                        pairs = zip(utf8_bytes[:-1], utf8_bytes[1:])
                        h = []
                        for index, pair in enumerate(pairs):
                            if pair in self.merges:
                                heapq.heappush(h, (self.merges_order[pair], index))
                        if len(h) > 0:
                            merge_index = h[0][1]
                            new_byte = utf8_bytes[merge_index] + utf8_bytes[merge_index+1]
                            utf8_bytes.pop(merge_index)
                            utf8_bytes.pop(merge_index)
                            utf8_bytes.insert(merge_index, new_byte)
                        else:
                            break
                    for word in utf8_bytes:
                        encode_result.append(self.vocab_token_to_id[word])
        return encode_result                        
                
        
    def encode_iterable(self, iterable: Iterable[str])-> Iterator[list[int]]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)
        
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text. """        
        return b''.join(self.vocab[token_id] for token_id in ids).decode('utf-8', errors='replace') 