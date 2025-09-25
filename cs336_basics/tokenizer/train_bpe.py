import regex as re
import os
import multiprocessing
from typing import BinaryIO

def split_by_special(text, special_tokens, drop_special=True):
    """
    Split text by special tokens. If drop_special=True, special tokens are removed.
    """
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: 
        pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]  # remove empty strings

def process_chunk(args):
    """
    Process a single text chunk for multiprocessing.
    
    Args:
        args (tuple): Contains (chunk_text, special_tokens, pattern_str)
        
    Returns:
        dict: Token counts for this chunk
    """
    chunk_text, special_tokens, pattern_str = args
    
    # Compile pattern in worker process
    pattern = re.compile(pattern_str)
    chunk_counts = {}
    
    # Split by special tokens and remove them (following bpe.py approach)
    text_chunks = split_by_special(chunk_text, special_tokens, drop_special=True)
    
    # Process each chunk (with special tokens removed)
    for text_chunk in text_chunks:
        matches = re.finditer(pattern, text_chunk)
        for match in matches:
            token = match.group(0)
            # Convert to tuple of bytes objects (like bpe.py)
            utf8_bytes = list(token.encode('utf-8'))
            token_bytes = tuple(bytes([i]) for i in utf8_bytes)
            if len(token_bytes) >= 2:  # Only consider tokens of length >= 2
                chunk_counts[token_bytes] = chunk_counts.get(token_bytes, 0) + 1
    
    return chunk_counts

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks : int, special_tokens : list[str]) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
    if special_tokens:
        split_special_token = "|".join(re.escape(token) for token in special_tokens)
        pattern = re.compile(split_special_token.encode('utf-8'))
    else:
        pattern = None
    #split_special_token_bytes = split_special_token.encode('utf-8')

    #print(split_special_token)
    for bi in range(1, len(chunk_boundaries) - 1):
        # Start searching from the previous boundary to avoid missing special tokens
        initial_position = chunk_boundaries[bi - 1]
        target_boundary = chunk_boundaries[bi]
        found_special_token = False
        
        file.seek(initial_position)
        
        while initial_position < file_size:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            if pattern:
                match = pattern.search(mini_chunk)
                if match:
                    found_position = initial_position + match.end()
                    if found_position <= target_boundary:
                        chunk_boundaries[bi] = found_position
                        found_special_token = True
                        break
            
            initial_position += mini_chunk_size
            
        # If no special token found, keep the original boundary
        if not found_special_token and initial_position >= file_size:
            chunk_boundaries[bi] = target_boundary
    
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(input_path : str, vocab : dict[bytes, int], vocab_size : int, special_tokens : list[str]):
    num_processes = 4

    # GPT-2 tokenization pattern (without special tokens since they're removed)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pretoken_counts = dict()

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunk_texts = split_by_special(text,special_tokens)
        
    # Prepare arguments for multiprocessing
    process_args = [(chunk_text, special_tokens, pattern) for chunk_text in chunk_texts]
        
    # Process chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, process_args)
        
    # Merge results from all processes
    for chunk_counts in chunk_results:
        for token_bytes, count in chunk_counts.items():
            pretoken_counts[token_bytes] = pretoken_counts.get(token_bytes, 0) + count
    
    return pretoken_counts



def train_bpe(input_path : str, vocab_size : int, special_tokens : list[str]):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given input corpus.
    
    This implementation follows the standard BPE algorithm where special tokens are
    removed from the training data to prevent them from being merged with other tokens.
    
    Args:
        input_path (str): Path to the input text file for training.
        vocab_size (int): Total number of items in the final vocabulary (including 
            special tokens and base byte tokens).
        special_tokens (list[str]): List of special tokens to be added to the vocabulary.
            These tokens are removed from training data and will not be merged.
            
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: A tuple containing:
            - vocab: The tokenizer vocabulary, a mapping from int (token ID) to bytes 
              (token bytes). Special tokens get IDs 0, 1, ..., len(special_tokens)-1,
              followed by byte tokens for values 0-255, then merged tokens.
            - merges: List of BPE merges produced from training. Each item is a tuple 
              of bytes (token1, token2), representing that token1 was merged with token2.
              Merges are ordered by creation time.
    """
    vocab = dict()
    special_token_index = 0
    for special_token in special_tokens:
        vocab[special_token_index] = special_token.encode('utf-8') # encode is used from str to bytes
        special_token_index += 1
    # Add byte tokens (0-255) to the vocabulary, starting after the special tokens
    vocab.update({i+special_token_index: bytes([i]) for i in range(256)})
    
    pretoken_counts = pretokenize(input_path, vocab, vocab_size, special_tokens)
    
    # compute bpe merges
    # Note: no need to skip special tokens since they are removed in pretokenize
    merges = list()
    while len(vocab) < vocab_size:
        pair_counts = dict()
        for token_bytes, count in pretoken_counts.items():
            for i in range(len(token_bytes) - 1):
                left = token_bytes[i]
                right = token_bytes[i+1]
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        
        if not pair_counts:
            break
        
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        
        new_pretoken_counts = dict()
        for token_bytes, count in pretoken_counts.items():    
            new_token_bytes = []
            i = 0
            while i < len(token_bytes):
                if i < len(token_bytes) - 1:
                    left = token_bytes[i]
                    right = token_bytes[i+1]
                    current_pair = (left, right)
                    if current_pair == best_pair:
                        new_token_bytes.append(new_token)
                        i += 2
                        continue
                
                new_token_bytes.append(token_bytes[i])
                i += 1
            new_token_tuple = tuple(new_token_bytes)
            if new_token_tuple:
                new_pretoken_counts[new_token_tuple] = new_pretoken_counts.get(new_token_tuple, 0) + count
        if not new_pretoken_counts:
            break
        pretoken_counts = new_pretoken_counts
    return vocab, merges

# input_path = "/home/joye/learning/cs336/assignment1-basics/tests/fixtures/corpus.en"

# vocab, merges = train_bpe(
#     input_path=input_path,
#     vocab_size=500,
#     special_tokens=["<|endoftext|>"],
# )

# for i, m in enumerate(merges):
#     print(i, m)