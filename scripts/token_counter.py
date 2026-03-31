"""
Token Counter - Counts the number of tokens in a text file using GPT-2 tokenizer.

Usage:
    python token_counter.py <input_file.txt>
    
Example:
    python token_counter.py my_text.txt
"""

import sys
from transformers import GPT2TokenizerFast


def count_tokens(file_path: str) -> int:
    """
    Count the number of GPT-2 tokens in a text file.
    
    Args:
        file_path: Path to the input text file.
        
    Returns:
        Number of tokens in the file.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    return len(tokens)


def main():
    if len(sys.argv) != 2:
        print("Usage: python token_counter.py <input_file.txt>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        token_count = count_tokens(file_path)
        print(f"Token count: {token_count}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
