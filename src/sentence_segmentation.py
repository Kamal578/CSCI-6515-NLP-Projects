import re

# Define common abbreviations that should not split sentences
ABBREVIATIONS = [
    "Dr.", "Mr.", "Mrs.", "Prof.", "A.M.", "S.A.", "B.C.", "M.A.", "Ph.D.",
    "U.S.", "e.g.", "i.e.", "etc.", 'prof.'
]

# Regular expressions to match punctuation marks and sentence enders
PUNCTUATION_MARKS = r"[.!?]"
GUILLEMET_OPEN = r"[“«]"
GUILLEMET_CLOSE = r"[”»]"
COLON = r":"

# Function to check if a string is an abbreviation
def is_abbreviation(token: str) -> bool:
    return token in ABBREVIATIONS

# Function to check if a period or comma is surrounded by non-space characters
def is_surrounded_by_non_space(text: str, i: int) -> bool:
    """
    Checks if the punctuation at index `i` is surrounded by non-space characters.
    If so, it's not a sentence boundary.
    """
    if i > 0 and i < len(text) - 1:
        return text[i-1] != " " and text[i+1] != " "
    return False

# Function to handle the segmentation of a given text
def sentence_segment(text: str) -> list:
    sentences = []
    start = 0
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]

        # NEW Rule: guillemet + space + Capital letter → sentence boundary
        if re.match(GUILLEMET_CLOSE, ch):
            if i + 2 < length and text[i + 1] == " " and text[i + 2].isupper():
                sentences.append(text[start:i + 1].strip())
                start = i + 1
                i += 1
                continue
        
        # Case 1: Look for sentence-ending punctuation marks (., !, ?)
        if re.match(PUNCTUATION_MARKS, ch):
            # If surrounded by non-space characters, it's not a sentence boundary
            if is_surrounded_by_non_space(text, i):
                i += 1
                continue
            
            # Look at the token before punctuation
            chunk = text[start:i+1]
            prev_token = chunk.rstrip().split()[-1]  # The last word before punctuation
            
            # Case 2: Handle abbreviations
            if is_abbreviation(prev_token):
                i += 1
                continue

            # NEW Case: Handle single-letter initials like "A.", "B."
            if ch == "." and i > 0 and text[i - 1].isupper():
                i += 1
                continue

            
            # Case 3: Handle periods in numbers like "3.14" (decimal numbers)
            # = if dot or comma surrounded by any non-space character - sentence continues
            if re.match(r"(?<=\S)[.,](?=\S)", prev_token):
                i += 1
                continue
            
            # Case 4: Handle initials like "A.M.", "S.B." where the period is not a sentence boundary
            if re.match(r"[A-Z]\.[A-Z]", prev_token):
                i += 1
                continue

            # Case 5: Handle punctuation inside quotes or guillemets
            if i + 1 < length and re.match(GUILLEMET_OPEN, text[i + 1]):
                # After a punctuation mark inside guillemets, check context
                if i + 2 < length and re.match(r"\s+[A-Z]", text[i + 2]):
                    # Sentence ends when a space and uppercase letter follow the guillemet
                    sentences.append(chunk.strip())
                    start = i + 1
                else:
                    i += 1
                    continue
            
            # Otherwise, mark the sentence boundary
            sentences.append(chunk.strip())
            start = i + 1
        
        # Case 6: Handle colons as non-sentence breakers
        elif re.match(COLON, ch):
            # Just skip over the colon
            i += 1
            continue
        
        i += 1

    # Final chunk (to capture the last sentence after the loop ends)
    if start < length:
        sentences.append(text[start:].strip())

    return sentences

# Example usage with the gold standard sentences
text= r'Salam. J.Epstein was in ADA! today "citation." and "another citation." New sentence? prof. Rustamov salam deyir. S. Rustamov salam deyir! S.Rustamov salam deyir!'


# Test the sentence segmentation algorithm
segmented_sentences = sentence_segment(text)
for sentence in segmented_sentences:
    print("--------\n"+sentence)

