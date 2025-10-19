import pandas as pd
import re
import unicodedata
from collections import Counter
from tqdm import tqdm
import os

# Register tqdm for pandas apply() progress bars
tqdm.pandas()

def clean_text(text):
    """
    Strips HTML tags and normalizes whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def canonicalize_brand(phrase):
    """
    Canonicalize a brand phrase by casefolding, removing punctuation/diacritics,
    collapsing whitespace, and trimming. Example: "L’Oréal" -> "loreal".
    """
    if not isinstance(phrase, str):
        return None
    s = phrase.strip()
    if not s:
        return None
    # Unicode normalize and strip accents
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    # Casefold
    s = s.casefold()
    # Normalize possessives BEFORE stripping punctuation so we keep the token joined
    # e.g., "member's" -> "members", "miller's" -> "millers"
    s = re.sub(r"\b([a-z0-9]+)'s\b", r"\1s", s)
    s = re.sub(r"\b([a-z0-9]+)s'\b", r"\1s", s)
    # Replace common punctuation with space, then strip others
    s = re.sub(r"[\u2019\u2018’]", "'", s)  # normalize apostrophes
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^\w\s]", " ", s)  # keep letters/digits/underscore/space
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def extract_title_portion(text):
    """
    Extract the title portion from catalog_content, focusing on the line after
    "Item Name:" and stopping at the first strong delimiter.
    """
    if not isinstance(text, str):
        return ""
    m = re.search(r"Item Name:\s*(.*)", text, flags=re.IGNORECASE)
    snippet = m.group(1) if m else text
    # Cut at first strong delimiter or parenthesis start
    parts = re.split(r"\s*(?:[\-\–\—\|,;/\n]|\(|\[|Pack of|Count|CT\b)\s*",
                     snippet, maxsplit=1, flags=re.IGNORECASE)
    title = parts[0].strip()
    return title

STRONG_PRODUCT_NOUNS = {
    # Food basics
    'sauce','taco','salsa','chips','cracker','crackers','cereal','candy','chocolate',
    'coffee','tea','oats','oatmeal','pasta','rice','beans','juice','soup','oil',
    'vinegar','salt','pepper','seasoning','spice','spices','flour','sugar','water',
    'soda','milk','yogurt','cheese','bread','bar','bars',
    # Beauty/household
    'shampoo','conditioner','soap','detergent','lotion','cream','toothpaste',
    'toothbrush','cleaner','wipes','deodorant','spray',
}

# Tokens that commonly appear as the SECOND token of true multi-word brands.
# If present, we will prefer at least a bigram brand (e.g., "Bear Creek").
BRAND_SUFFIX_SECOND_TOKENS = {
    'creek', 'valley', 'farms', 'farm', 'grove', 'gardens', 'garden', 'mills',
    'market', 'roasters', 'brew', 'brewing', 'brewery', 'house', 'river',
    'mountain', 'mountains', 'coffee', 'tea', 'foods', 'organics', 'kitchen',
    'kitchens', 'country', 'company', 'co'
}

# Honorific-like first tokens that should not stand alone as brands
HONORIFIC_PREFIXES = { 'mr', 'mrs', 'ms', 'dr', 'st', 'saint' }

# Trailing generic tokens to trim from the END of a detected brand. This helps
# reduce forms like "bear creek country kitchens" to "bear creek".
BRAND_TAIL_TOKENS = {
    'company', 'co', 'inc', 'incorporated', 'ltd', 'limited', 'llc', 'corp',
    'corporation', 'brand', 'brands', 'products', 'product', 'country',
    'kitchen', 'kitchens', 'food', 'foods', 'market', 'markets'
}

def leading_proper_noun_tokens(tokens):
    """
    Return the leading span of tokens that look like proper nouns or brand tokens.
    Heuristic: start-of-title tokens that are TitleCase/ALLCAPS/contain brand-ish chars.
    """
    span = []
    for tok in tokens:
        if not tok:
            break
        # Strip surrounding punctuation for the check
        core = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
        if not core:
            break
        # Proper if starts with uppercase or is all-caps alnum, allow apostrophes/dots/&/hyphen in original
        if re.match(r"^[A-Z][\w\-\.']*$", core) or re.match(r"^[A-Z0-9]{2,}$", core):
            span.append(tok)
            continue
        # Allow small connectors like "&" or "and" inside a proper span
        if core in {"&", "and", "n"}:
            span.append(tok)
            continue
        break
    return span

def pre_noun_tokens(title):
    """
    Split title and return tokens before the first strong product noun (if any).
    """
    tokens = title.split()
    # Find first strong product noun boundary
    boundary = None
    for idx, tok in enumerate(tokens):
        if tok and tok.strip():
            t = re.sub(r"[^A-Za-z]", "", tok).lower()
            if t in STRONG_PRODUCT_NOUNS:
                boundary = idx
                break
    if boundary is not None:
        tokens = tokens[:boundary]
    return tokens

def generate_prefixes(tokens, max_len=4):
    """
    Generate all leading prefixes up to max_len from a token list.
    """
    prefixes = []
    upto = min(len(tokens), max_len)
    # Avoid 1-token prefixes for common leading function words (e.g., "La", "The")
    stop_first_tokens = {"la","le","el","los","las","the","a","an","de","del","da","di"}
    start_k = 1
    if upto >= 2 and tokens and tokens[0].lower() in stop_first_tokens:
        start_k = 2
    for k in range(start_k, upto + 1):
        prefixes.append(" ".join(tokens[:k]))
    return prefixes

def build_brand_lexicon(df):
    """
    Build a frequency dictionary of plausible brand n-grams from train titles.
    Uses leading proper-noun spans before strong product nouns, then counts all
    leading prefixes up to length 4 (canonicalized).
    """
    counts = Counter()
    for text in tqdm(df['catalog_content'], desc='Building brand lexicon'):
        title = extract_title_portion(clean_text(text))
        if not title:
            continue
        tokens = pre_noun_tokens(title)
        tokens = leading_proper_noun_tokens(tokens)
        if not tokens:
            continue
        for prefix in generate_prefixes(tokens, max_len=4):
            canon = canonicalize_brand(prefix)
            if not canon:
                continue
            # Count the raw canonical form
            counts[canon] += 1
            # Also count a trimmed form without trailing generic tokens to
            # strengthen multi-word brand cores like "bear creek".
            trimmed = trim_brand_tail_tokens(canon)
            if trimmed and trimmed != canon:
                counts[trimmed] += 1
    return counts

def extract_brand_from_text(text, brand_freq):
    """
    Extract brand using a frequency-backed selection of leading proper-noun
    prefixes. Returns canonicalized brand string or None when absent/uncertain.
    Selection rule:
      - Generate prefixes (1..4 tokens) from leading proper-noun span before
        the first strong product noun.
      - Choose the prefix with the highest corpus frequency; if a longer
        prefix does not beat the shorter by a margin (e.g., 1.25x), prefer the
        shorter prefix to avoid including variants like colors/flavors.
    """
    if not isinstance(text, str):
        return None
    title = extract_title_portion(clean_text(text))
    if not title:
        return None
    tokens = pre_noun_tokens(title)
    tokens = leading_proper_noun_tokens(tokens)
    if not tokens:
        return None
    prefixes = generate_prefixes(tokens, max_len=4)
    if not prefixes:
        return None

    # Enforce minimum brand length (bigram) in common cases where single-token
    # detection would be wrong: honorific first tokens or possessive first token,
    # and when the second token is a typical brand suffix (e.g., creek, farms).
    t0_core = re.sub(r"[^A-Za-z]", "", tokens[0]).lower() if tokens else ""
    first_is_honorific = t0_core in HONORIFIC_PREFIXES
    first_is_possessive = bool(re.search(r"[’']s\b", tokens[0]))
    second_is_suffix = len(tokens) >= 2 and tokens[1].lower() in BRAND_SUFFIX_SECOND_TOKENS
    min_len = 2 if (first_is_honorific or first_is_possessive or second_is_suffix) else 1

    # Score prefixes by frequency (consider both raw and trimmed variants)
    scored = []
    for p in prefixes:
        num_tokens = len(p.split())
        if num_tokens < min_len:
            continue
        c = canonicalize_brand(p)
        if not c:
            continue
        c_trim = trim_brand_tail_tokens(c)
        freq = max(brand_freq.get(c, 0), brand_freq.get(c_trim, 0))
        scored.append((p, c_trim or c, freq, num_tokens))
    if not scored:
        return None

    # Pick best by frequency; prefer LONGER when frequencies are close.
    # Sort: higher freq first, then longer token count, then lexical length.
    scored.sort(key=lambda x: (-x[2], -x[3], -len(x[1])))
    best_phrase, best_canon, best_freq, _ = scored[0]

    # Safety: if best ends with generic tails, trim again.
    best_canon = trim_brand_tail_tokens(best_canon)
    return best_canon or None

def trim_brand_tail_tokens(canon: str) -> str:
    """Trim trailing generic tokens like "company", "kitchens".

    Examples:
      "bear creek country kitchens" -> "bear creek"
    """
    if not canon:
        return canon
    toks = canon.split()
    while toks and toks[-1] in BRAND_TAIL_TOKENS:
        toks.pop()
    return " ".join(toks) if toks else canon

def parse_units_and_quantity(text):
    """
    Detects numeric quantities, normalizes them to base units, and computes derived totals.
    """
    text_lower = text.lower()
    
    # --- Normalize to base units ---
    mass_conversions = {'oz': 28.35, 'ounce': 28.35, 'lb': 453.59, 'pound': 453.59, 'g': 1, 'gram': 1, 'kg': 1000}
    vol_conversions = {'fl oz': 29.5735, 'ml': 1, 'l': 1000, 'liter': 1000}
    count_conversions = {'ct': 1, 'count': 1, 'pcs': 1, 'pack': 1}

    total_mass_g = 0.0
    total_volume_ml = 0.0
    total_count_each = 0.0

    # --- Detect flexible patterns ---
    # Patterns like "11 oz", "1.6 oz, 16 ct", "16 oz x 12"
    patterns = re.findall(r'(\d+(?:\.\d+)?)\s*(fl oz|oz|ounce|lb|pound|g|gram|kg|ml|l|liter|ct|count|pcs|pack)\b', text_lower)
    
    # First pass: sum up individual unit mentions
    for value, unit in patterns:
        val = float(value)
        if unit in mass_conversions:
            # Distinguish mass oz from fluid fl oz
            if unit == 'oz' and 'fl oz' in text_lower:
                continue # Avoid double counting if 'fl oz' is also present
            total_mass_g += val * mass_conversions[unit]
        elif unit in vol_conversions:
            total_volume_ml += val * vol_conversions[unit]
        elif unit in count_conversions:
            total_count_each += val

    # --- Handle multipliers like "(Pack of 6)" or "x 12" ---
    multiplier = 1.0
    multiplier_match = re.search(r'(?:pack of|case of|x|×)\s*(\d+)', text_lower)
    if multiplier_match:
        multiplier = float(multiplier_match.group(1))

    # Apply multiplier. If a count was already parsed, the multiplier might be redundant.
    # Logic: If we found a specific count (e.g., "90 ct"), the multiplier applies to mass/volume of the pack.
    # If no specific count was found, the multiplier IS the count.
    if multiplier > 1:
        if total_count_each == 0:
            total_count_each = multiplier
        if total_mass_g > 0:
            total_mass_g *= multiplier
        if total_volume_ml > 0:
            total_volume_ml *= multiplier
            
    # Final check: every product is at least one count
    if total_count_each == 0:
        total_count_each = 1.0

    return pd.Series([total_mass_g, total_volume_ml, total_count_each])

def extract_keyword_flags(text):
    """
    Creates binary flags for high-signal keywords and category cues.
    """
    text_lower = text.lower()
    
    keywords = [
        'organic', 'vegan', 'kosher', 'gluten free', 'keto', 'imported', 
        'premium', 'gourmet', 'refill', 'bundle', 'case', 'hot sauce', 
        'cereal', 'tea', 'coffee', 'candy', 'spice', 'cosmetics', 'cleaning'
    ]
    
    flags = {}
    for keyword in keywords:
        col_name = f'flag_{keyword.replace(" ", "_")}'
        flags[col_name] = 1 if keyword in text_lower else 0
        
    return pd.Series(flags)

def process_text_features(df, brand_freq):
    """
    Main function to orchestrate the text processing and feature engineering pipeline.
    """
    # Create a working copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # --- 6.1 Cleaning ---
    # Preserve original case for brand extraction
    df_processed['cleaned_content_orig_case'] = df_processed['catalog_content'].progress_apply(clean_text)
    # Create a lower-cased version for keyword/regex matching
    df_processed['cleaned_content_lower'] = df_processed['cleaned_content_orig_case'].str.lower()
    
    # --- 6.2 Brand Extraction ---
    print("Extracting brands with lexicon...")
    df_processed['brand'] = df_processed['cleaned_content_orig_case'].progress_apply(
        lambda s: extract_brand_from_text(s, brand_freq)
    )

    # --- 6.3 Quantity, Units, and Pack Parsing ---
    print("Parsing units and quantities...")
    unit_features = df_processed['cleaned_content_lower'].progress_apply(parse_units_and_quantity)
    unit_features.columns = ['total_mass_g', 'total_volume_ml', 'total_count_each'] #
    df_processed = pd.concat([df_processed, unit_features], axis=1)

    # --- 6.4 Claims and Category Cues ---
    print("Extracting keyword flags...")
    keyword_flags = df_processed['cleaned_content_lower'].progress_apply(extract_keyword_flags)
    df_processed = pd.concat([df_processed, keyword_flags], axis=1)
    
    # Drop intermediate columns
    df_processed.drop(columns=['cleaned_content_orig_case', 'cleaned_content_lower'], inplace=True)
    
    return df_processed

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        print("Loading raw data...")
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')

        print("\n--- Building Brand Lexicon from Training Data ---")
        brand_lexicon = build_brand_lexicon(train_df)
        print(f"Brand lexicon size: {len(brand_lexicon)}")

        print("\n--- Processing Training Data ---")
        train_features_df = process_text_features(train_df, brand_lexicon)
        
        print("\n--- Processing Test Data ---")
        test_features_df = process_text_features(test_df, brand_lexicon)
        
        output_train_path = 'dataset/train_features.csv'
        output_test_path = 'dataset/test_features.csv'
        
        train_features_df.to_csv(output_train_path, index=False)
        test_features_df.to_csv(output_test_path, index=False)

        print(f"\n✅ Feature engineering complete.")
        print(f"Training features saved to: {output_train_path}")
        print(f"Test features saved to: {output_test_path}")
        
        print("\nSample of the generated training data:")
        print(train_features_df.head())

    except FileNotFoundError as e:
        print(f"\nError: Data file not found. {e}")
        print("Please ensure 'dataset/train.csv' and 'dataset/test.csv' are in the correct directory.")