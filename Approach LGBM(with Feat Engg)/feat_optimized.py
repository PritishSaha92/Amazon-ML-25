# GH200-OPTIMIZED Feature Extraction Script
# Maximizes throughput with pre-loading, larger batches, and pipelining

import os
import pandas as pd
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import gc
import queue
import threading
from collections import deque

# ============================================================================
# 🚀 GH200-OPTIMIZED CONFIGURATION
# ============================================================================
DATASET_MODE = 'train'  # Change to 'test' when ready

if DATASET_MODE == 'train':
    CSV_FILE_PATH = 'dataset/train.csv'
    IMAGES_DIR = 'images/train/'
    OUTPUT_CSV_FILE = 'output/vlm_structured_data_train.csv'
    STREAM_OUT_CSV_FILE = 'output/vlm_structured_data_stream_train.csv'
else:
    CSV_FILE_PATH = 'dataset/test.csv'
    IMAGES_DIR = 'images/test/'
    OUTPUT_CSV_FILE = 'output/vlm_structured_data_test.csv'
    STREAM_OUT_CSV_FILE = 'output/vlm_structured_data_stream_test.csv'

OUTPUT_DIR = 'output/'
VLM_DEBUG_PATH = os.path.join(OUTPUT_DIR, f'vlm_debug_raw_{DATASET_MODE}.txt')

# 🔥 AGGRESSIVE GH200 SETTINGS (96GB VRAM + 64 vCPUs + 432GB RAM)
BATCH_SIZE = 3072  # MASSIVE outer batch (you have 432GB RAM!)
MAX_MICRO_BATCH_SIZE = 384  # Large micro-batch (with 8-12% usage, we can go 3-4x!)
MAX_NEW_TOKENS = 80
MAX_IMAGE_SIDE = 384  # Smaller for speed
MAX_INPUT_CHARS = 1500
STREAM_EVERY_N = 1000

# Image pre-loading pipeline
NUM_PRELOAD_WORKERS = 60  # 60/64 vCPUs for I/O (leave 4 for system)
IMAGE_PREFETCH_BATCHES = 5  # Keep 5 batches ahead in RAM
MAX_QUEUE_SIZE = 10  # Max batches in memory

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 42

print(f"🎯 DATASET MODE: {DATASET_MODE.upper()}")
print(f"📁 Input CSV: {CSV_FILE_PATH}")
print(f"🚀 GH200-OPTIMIZED: BATCH={BATCH_SIZE}, MICRO_BATCH={MAX_MICRO_BATCH_SIZE}, WORKERS={NUM_PRELOAD_WORKERS}")

# ============================================================================
# Image Pre-loading Pipeline (Eliminates GPU Stalls!)
# ============================================================================

class ImagePreloader:
    """Asynchronously pre-loads and preprocesses images to eliminate I/O bottleneck"""
    
    def __init__(self, image_paths, texts, max_workers=56, prefetch_batches=5, batch_size=3072, max_side=384):
        self.image_paths = image_paths
        self.texts = texts
        self.max_workers = max_workers
        self.prefetch_batches = prefetch_batches
        self.batch_size = batch_size
        self.max_side = max_side
        self.queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.total_batches = (len(image_paths) + batch_size - 1) // batch_size
        self.stop_event = threading.Event()
        self.loader_thread = None
        
    def _resize_image(self, path):
        """Load and resize single image"""
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if max(w, h) > self.max_side:
                scale = self.max_side / float(max(w, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            return img
        except Exception as e:
            # Return black image on error
            return Image.new("RGB", (self.max_side, self.max_side), color=(0, 0, 0))
    
    def _load_batch(self, start_idx, end_idx):
        """Load one batch of images in parallel"""
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_texts = self.texts[start_idx:end_idx]
        
        # Parallel image loading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            images = list(executor.map(self._resize_image, batch_paths))
        
        return {
            'images': images,
            'texts': batch_texts,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
    
    def _loader_worker(self):
        """Background thread that continuously loads batches ahead"""
        for batch_idx in range(self.total_batches):
            if self.stop_event.is_set():
                break
                
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.image_paths))
            
            batch_data = self._load_batch(start_idx, end_idx)
            self.queue.put(batch_data)  # Blocks if queue is full
            
        # Signal completion
        self.queue.put(None)
    
    def start(self):
        """Start background loading"""
        self.loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.loader_thread.start()
    
    def get_batch(self):
        """Get next pre-loaded batch (blocks until ready)"""
        batch = self.queue.get()
        return batch
    
    def stop(self):
        """Stop background loading"""
        self.stop_event.set()
        if self.loader_thread:
            self.loader_thread.join(timeout=5)


# ============================================================================
# Setup
# ============================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Setting up Qwen2.5-VL...")
device = "cuda"
print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0)})")
torch.cuda.empty_cache()

# Performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:
    torch.nn.attention.sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except:
    pass
torch.manual_seed(SEED)

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'left'
processor.tokenizer.padding_side = 'left'

model.eval()
print("✅ Model loaded")

# Warmup
dummy_img = Image.new("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), color=(0, 0, 0))
warm_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "warmup"}]}]
warm_text = processor.apply_chat_template(warm_messages, add_generation_prompt=True, tokenize=False)
tpl = processor(text=[warm_text], images=[dummy_img], return_tensors="pt", padding=True)
tpl = {k: v.to(device, non_blocking=True) for k, v in tpl.items()}
_ = model.generate(**tpl, max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True)
del tpl
torch.cuda.empty_cache()
print("🔥 Warmup complete")

# ============================================================================
# System Prompt (keeping your existing one)
# ============================================================================

KEYWORD_LIST = [
    'organic', 'vegan', 'kosher', 'gluten free', 'keto', 'imported',
    'premium', 'gourmet', 'refill', 'bundle', 'case', 'exotic', 'luxury',
    'natural', 'sugar free', 'low fat', 'plant-based', 'whole grain'
]

CATEGORIES = [
    'Beverages', 'Grains & Pasta', 'Snacks & Sweets', 'Condiments, Sauces & Spices',
    'Canned & Packaged Goods', 'Baking Supplies', 'Dairy & Dairy Alternatives',
    'Breakfast Foods', 'Oils, Vinegars & Dressings', 'Produce (Packaged & Dried)',
    'Meat & Seafood (Shelf-Stable)', 'Health & Wellness Foods', 'Baby Food',
    'Frozen Foods', 'International & Gourmet Foods',
]
CATEGORY_FALLBACK = 'Uncategorized'

KEYWORD_LIST_STR = ', '.join(KEYWORD_LIST)
CATEGORIES_STR = ', '.join(CATEGORIES)

SYSTEM_PROMPT = f"""Extract product information from the image and text. Output EXACTLY 6 lines:

brand: <name>
pack_count: <number>
mass_g: <grams>
volume_ml: <milliliters>
keywords: <list>
category: <name>

CRITICAL - BRAND NAME ONLY:
The brand is ONLY the company name, NOT the product description.
- Look at the logo or brand name on the package (usually 1-3 words max)
- STOP at product type words: sauce, soup, salt, chips, cereal, drink, tea, nuts, tuna, etc.
- STOP at flavors: mild, spicy, original, classic, etc.
- STOP at descriptors: hearty, creamy, fine, pink, green, etc.
- STOP at any numbers or units: 200GM, 2 oz, 59 mL, 1 lb, Pack of 6

RULES:
- Brand: 1-3 words max, company name only
- Numbers: Just numbers, no units
- Conversions: oz×28.35=g, lb×453.59=g, fl oz×29.57=ml  
- Pack count: COUNT items visible OR "Pack of X"
- Keywords: ONLY from: {KEYWORD_LIST_STR}
- Category: ONE from: {CATEGORIES_STR}"""

# ============================================================================
# Processing Function with FULL fallback logic from notebook
# ============================================================================

# Helper functions for robust parsing & fallbacks
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def uncamelcase(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

def normalize_brand_text(brand_text: str) -> str:
    b = normalize_whitespace(brand_text.replace("'", "'").replace('"', '"').replace('"', '"'))
    b = re.sub(r"\s*(?:pack[_ ]?count|mass[_ ]?g|volume[_ ]?ml|keywords|category)\s*[:|-].*$", "", b, flags=re.IGNORECASE)
    b = uncamelcase(b)
    b = re.sub(r"([A-Za-z])([A-Z][a-z])", r"\1 \2", b)  # BearCreek -> Bear Creek
    return b[:64].strip() if b else "Unknown"

def extract_brand_from_catalog(text: str) -> str:
    """Extract ONLY the brand name from catalog text"""
    if not text:
        return None
    
    m = re.search(r"Item\s*Name:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    candidate = m.group(1).strip() if m else next((ln.strip() for ln in text.split('\n') if ln.strip()), "")
    candidate = re.split(r"\s*,\s*|\s*[\(/\[]\s*|\s+-\s+", candidate, maxsplit=1)[0].strip()
    
    stop_words = {
        'sauce','salsa','dressing','marinade','dip','spread','soup','stew','chili','broth','stock','bowl',
        'salt','pepper','spice','seasoning','herb','bouillon','rub','marinade','powder','blend',
        'chip','chips','crisps','cracker','crackers','pretzel','pretzels','cookie','cookies','snack','snacks',
        'cereal','granola','oats','oatmeal','bar','bars','candy','chocolate','gummies','gummi','sweets',
        'coffee','tea','drink','beverage','juice','soda','water','gooseberry','amla',
        'pasta','noodles','rice','beans','grains','fruit','nuts','mix','trail','tuna','shrimp','fish','jerky',
        'oil','vinegar','syrup','honey','flour','sugar','baking','cheese','butter','milk','yogurt','pickle','pickles',
        'organic','natural','premium','gourmet','original','classic','mild','medium','hot','spicy','sweet','tangy','zesty','chatpata',
        'hearty','creamy','chunky','smooth','crispy','crunchy','fine','coarse','ground','whole','sliced','diced',
        'black','white','red','pink','green','golden','yellow','light','lite','diet','low','reduced','zero','fat-free','sugar-free',
        'plus','max','ultra','extra','super','thirst','quencher','energy','sport','sports','chunk','solid','albacore',
        'women','women\'s','mens','men\'s','kids','children','baby'
    }
    units = {'oz','ounce','ounces','ml','g','gram','grams','kg','lb','pound','fl','floz','fl-oz','liter','litre','l'}
    ingredient_words = {
        'onion','garlic','paprika','turmeric','cumin','coriander','ginger','cinnamon','oregano','basil','thyme','rosemary','peppercorn',
        'pumpkin','chamomile','cheesecake'
    }
    
    words = candidate.split()
    brand_words = []
    for i, raw in enumerate(words):
        w = re.sub(r"[^\w&'+-]", '', raw)
        wl = w.lower()
        if any(ch.isdigit() for ch in wl):
            break
        if wl in units:
            break
        if wl in {'women','women\'s','mens','men\'s','kids','children','baby'}:
            if len(brand_words) == 0:
                continue
            else:
                break
        if wl in stop_words and '-' not in w:
            if len(brand_words) == 1 and len(wl) <= 8 and len(brand_words[0]) <= 8:
                brand_words.append(w)
                break
            else:
                break
        if wl in ingredient_words and len(brand_words) >= 1:
            break
        if wl == 'sea' and i + 1 < len(words) and words[i+1].lower().strip(':;,./-') == 'salt':
            break
        brand_words.append(w)
    
    if not brand_words:
        return None
    
    candidate = ' '.join(brand_words).strip().rstrip(' -–—,:;|/\\')
    candidate = normalize_brand_text(candidate)
    if len(candidate.split()) > 4:
        candidate = ' '.join(candidate.split()[:3])
    return candidate if len(candidate) >= 2 else None

# Pack count from catalog
WORD2NUM = {
    "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
    "eleven":11, "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15, "twenty":20
}

def parse_pack_count_from_catalog(text: str):
    if not text:
        return None
    def _word_or_num(s: str):
        s = s.lower().strip()
        if s.isdigit():
            return float(s)
        return float(WORD2NUM[s]) if s in WORD2NUM else None
    
    pats = [
        r"pack\s*of\s*([A-Za-z0-9]+)",
        r"\b([A-Za-z0-9]+)\s*pack\b",
        r"\b([A-Za-z0-9]+)\s*[- ]?count\b",
        r"\b([0-9]+)\s*x\s*[0-9]+(?:\.[0-9]+)?\s*(?:oz|ounce|fl\s*oz|ml|g|gram|lb|pound)s?\b",
        r"case\s*of\s*([A-Za-z0-9]+)",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            n = _word_or_num(m.group(1))
            if n is not None:
                return n
    return None

# Mass/volume from catalog
def parse_value_unit_from_catalog(text: str):
    if not text:
        return (None, None)
    mv = re.search(r"\bValue:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    mu = re.search(r"\bUnit:\s*([A-Za-z ]+)", text, re.IGNORECASE)
    if mv and mu:
        try:
            return float(mv.group(1)), mu.group(1).strip().lower()
        except:
            return (None, None)
    return (None, None)

def convert_to_mass_volume(value: float, unit: str):
    mass_g = 0.0
    volume_ml = 0.0
    if unit is None:
        return mass_g, volume_ml
    u = unit.lower().replace(".", "").replace("fluid", "fl").strip()
    if re.search(r"^(?:oz|ounce|ounces)$", u):
        mass_g = value * 28.35
    elif re.search(r"^(?:lb|pound|pounds)$", u):
        mass_g = value * 453.59
    elif re.search(r"^(?:g|gram|grams)$", u):
        mass_g = value
    elif re.search(r"^(?:fl\s*oz|floz)$", u):
        volume_ml = value * 29.57
    elif re.search(r"^(?:ml|milliliter|millilitre|milliliters|millilitres)$", u):
        volume_ml = value
    return float(mass_g), float(volume_ml)

# Keywords from catalog
def derive_keywords_from_text(text: str):
    if not text:
        return []
    t = text.lower()
    found = []
    synonyms = {
        'gluten free': ["gluten-free", "gluten free"],
        'sugar free': ["sugar-free", "sugar free", "no sugar"],
        'plant-based': ["plant based", "plant-based"],
        'whole grain': ["wholegrain", "whole grain"],
        'low fat': ["low-fat", "low fat"],
    }
    for kw in KEYWORD_LIST:
        if kw in synonyms:
            if any(s in t for s in synonyms[kw]):
                found.append(kw)
        else:
            if kw in t:
                found.append(kw)
    # de-dup
    seen, out = set(), []
    for k in found:
        if k not in seen:
            out.append(k); seen.add(k)
    return out

# Category from catalog
def categorize_from_text(text: str) -> str:
    if not text:
        return CATEGORY_FALLBACK
    t = text.lower()
    
    if any(b in t for b in ['pringles','fritos','cheetos','doritos','ruffles','lay\'s','lays']):
        return 'Snacks & Sweets'
    if any(b in t for b in ['starkist','bumble bee','chicken of the sea']):
        return 'Meat & Seafood (Shelf-Stable)'
    
    rules = [
        (['coffee','espresso','latte','cappuccino','k-cup','kcup','k cup','instant coffee','creamer','espresso capsules','k-cups','pods'], 'Beverages'),
        (['tea','herbal tea','green tea','iced tea','chai','jasmine','oolong','rooibos','yerba mate','matcha'], 'Beverages'),
        (['water','tonic','sparkling','seltzer','mineral water','alkaline water'], 'Beverages'),
        (['juice','nectar','cider','lemonade','smoothie'], 'Beverages'),
        (['soda','cola','pop','soft drink','energy drink','sports drink'], 'Beverages'),
        (['pretzel','chips','chip','crisps','candy','gummi','gummy','chocolate','cookie','cracker','popcorn','trail mix','nuts','peanuts','almonds','cashews','hazelnuts','pecans','pistachios','walnuts','snack','gum','mints','lollipop','taffy','fudge'], 'Snacks & Sweets'),
        (['ketchup','mustard','mayo','mayonnaise','hot sauce','sauce','spice','seasoning','cumin','pepper','salt','herb','bouillon','rub','marinade','pesto','dressing','vinaigrette'], 'Condiments, Sauces & Spices'),
        (['bean','beans','soup','noodle','pouch','can','canned','packaged','pasta sauce','fruit cocktail','pineapple','cherry','mandarin','olives','pickles','jelly','jam','preserves','jar'], 'Canned & Packaged Goods'),
        (['cereal','granola','oatmeal','oats','breakfast','toast','toaster pastry','bars'], 'Breakfast Foods'),
        (['pasta','rice','quinoa','noodles','grain','grains','fussili','spaghetti','macaroni'], 'Grains & Pasta'),
        (['flour','baking','yeast','baking powder','baking soda','icing','frosting','cake mix','brownie mix','cornmeal','breadcrumbs','bread crumbs'], 'Baking Supplies'),
        (['oil','olive oil','canola oil','vinegar','dressing','balsamic','avocado oil','sesame oil'], 'Oils, Vinegars & Dressings'),
        (['dried fruit','raisins','dates','nori','seaweed','dehydrated','freeze-dried','gooseberry','amla'], 'Produce (Packaged & Dried)'),
        (['tuna','sardine','salmon','anchovy','jerky','meat','chicken','shrimp','fish','pate'], 'Meat & Seafood (Shelf-Stable)'),
        (['protein','keto','vitamin','supplement','wellness','minoxidil','rogaine','omega','collagen','electrolyte'], 'Health & Wellness Foods'),
        (['baby food','infant','toddler','lil\' bits','puffs'], 'Baby Food'),
        (['frozen'], 'Frozen Foods'),
        (['imported','gourmet','italian','mexican','thai','indian','japanese','korean','international'], 'International & Gourmet Foods'),
        (['mason jar','wide mouth','capsules','toothpaste','mouthwash','soap','sunscreen'], 'Canned & Packaged Goods'),
        (['pet food','dog food','cat food','treats'], 'Canned & Packaged Goods'),
    ]
    for words, cat in rules:
        if any(w in t for w in words):
            return cat
    return CATEGORY_FALLBACK

def is_valid_brand(brand_text: str) -> bool:
    """Minimal guardrail: reject only obvious garbage"""
    if not brand_text or len(brand_text.strip()) == 0:
        return False
    b = brand_text.strip().lower()
    
    # Reject field names and common garbage
    invalid_patterns = [
        r'^brand\s*$', r'^pack[_ ]?count\s*[,:]*$', r'^mass[_ ]?g\s*[,:]*$', r'^volume[_ ]?ml\s*[,:]*$',
        r'^keywords\s*[,:]*$', r'^category\s*[,:]*$', r'^pack\s*[,:]*$', r'^<.*>$', r'^\d+$',
        r'^float\s*$', r'^str\s*$', r'^unknown\s*[,:]*$', r'^[.]+$', r'^[-]+$', r'^[,]+$',
        r'pack[_ ]count.*mass[_ ]g', r'mass[_ ]g.*volume', r'keywords.*category'
    ]
    if any(re.match(pat, b) for pat in invalid_patterns):
        return False
    
    # Reject standalone packaging/quantity words and common truncations (common VLM errors)
    if b in {'pack', 'packs', 'case', 'cases', 'count', 'box', 'boxes', 'bag', 'bags', 'bottle', 'bottles', 'jar', 'jars', 'can', 'cans', 'wood', 'organic', 'raw', 'natural', 'premium', 'fresh', 'classic', 'original'}:
        return False
    
    # Reject if contains measurements with numbers
    if re.search(r'\b\d+\s*(oz|ounce|ml|g|gram|kg|lb|pound)\b', b):
        return False
    
    # Reject if contains pack/case patterns
    if re.search(r'\b(pack|case)\s*(of|count)\b', b):
        return False
    
    # Reject if too long
    if len(brand_text.split()) > 5:
        return False
    
    # Reject if contains multiple field names
    field_count = sum(1 for field in ['pack_count', 'mass_g', 'volume_ml', 'keywords', 'category'] if field in b)
    if field_count >= 2:
        return False
    
    return True

def parse_vlm_response(response_text: str, catalog_text: str):
    """Parse VLM response with catalog fallback"""
    default = {"brand": "Unknown", "pack_count": 1.0, "mass": 0.0, "volume": 0.0, "keywords": [], "category": CATEGORY_FALLBACK}
    
    try:
        label_join = r"(?:brand|pack_count|mass[_ ]?g|volume[_ ]?ml|keywords|category)"
        # Brand: capture until newline, but stop at extra colons (malformed output like "Wood: 1")
        brand_match = re.search(r"^\s*brand\s*[:|-]\s*([^:\n]+?)(?:\s*[:]\s*[0-9])?(?=\n|$)", response_text, re.IGNORECASE | re.MULTILINE)
        pack_match = re.search(r"^\s*pack_count\s*[:|-]\s*([0-9]+(?:\.[0-9]+)?)\b", response_text, re.IGNORECASE | re.MULTILINE)
        mass_match = re.search(r"^\s*(?:mass|weight|net[\s_-]*weight)[\s_]*g\s*[:|-]\s*([0-9]+(?:\.[0-9]+)?)\b", response_text, re.IGNORECASE | re.MULTILINE)
        vol_match  = re.search(r"^\s*(?:volume|vol)[\s_]*ml\s*[:|-]\s*([0-9]+(?:\.[0-9]+)?)\b", response_text, re.IGNORECASE | re.MULTILINE)
        keywords_match  = re.search(rf"^\s*keywords\s*[:|-]\s*(.+?)(?=\n\s*{label_join}\s*[:|-]|$)", response_text, re.IGNORECASE | re.MULTILINE)
        category_match  = re.search(rf"^\s*category\s*[:|-]\s*(.+?)(?=\n\s*{label_join}\s*[:|-]|$)", response_text, re.IGNORECASE | re.MULTILINE)
        
        # Extract brand
        brand = "Unknown"
        if brand_match:
            brand_candidate = normalize_brand_text(brand_match.group(1))
            if is_valid_brand(brand_candidate):
                brand = brand_candidate
        
        # Extract numbers
        try:
            pack_count = float(pack_match.group(1)) if pack_match else 1.0
            if pack_count < 0.1 or pack_count > 10000:
                pack_count = 1.0
        except:
            pack_count = 1.0
        try:
            mass_val = float(mass_match.group(1)) if mass_match else 0.0
            if mass_val < 0 or mass_val > 100000:
                mass_val = 0.0
        except:
            mass_val = 0.0
        try:
            volume_val = float(vol_match.group(1)) if vol_match else 0.0
            if volume_val < 0 or volume_val > 50000:
                volume_val = 0.0
        except:
            volume_val = 0.0
        
        # Parse keywords
        raw_keywords = []
        if keywords_match:
            kw_text = keywords_match.group(1).strip()
            if kw_text.lower() not in ("", "none", "n/a", "null", "0", "0.0"):
                raw_keywords = [kw.strip().lower() for kw in kw_text.split(',') if kw.strip()]
        lower_to_canon = {k.lower(): k for k in KEYWORD_LIST}
        canonical_keywords = []
        seen = set()
        for k in raw_keywords:
            if k in lower_to_canon and k not in seen:
                canonical_keywords.append(lower_to_canon[k])
                seen.add(k)
        
        # Parse category
        category = CATEGORY_FALLBACK
        if category_match:
            cat_candidate = category_match.group(1).strip()
            if cat_candidate and cat_candidate.lower() not in ("", "none", "n/a", "null", "uncategorized"):
                if cat_candidate in CATEGORIES:
                    category = cat_candidate
                else:
                    lower_to_canon_cat = {c.lower(): c for c in CATEGORIES}
                    category = lower_to_canon_cat.get(cat_candidate.lower(), CATEGORY_FALLBACK)
        
        # Build result with VLM values
        clean = {
            "brand": brand,
            "pack_count": pack_count,
            "mass": mass_val,
            "volume": volume_val,
            "keywords": canonical_keywords,
            "category": category,
        }
        
        # Apply catalog fallbacks
        if catalog_text:
            # Brand fallback
            vlm_brand = clean["brand"]
            should_check_catalog = (
                vlm_brand == "Unknown" or 
                (len(vlm_brand.split()) <= 2 and not vlm_brand.lower().startswith(('mc', 'dr', 'st')))
            )
            if should_check_catalog:
                fb = extract_brand_from_catalog(catalog_text)
                if fb and fb != "Unknown":
                    if len(fb.split()) > len(vlm_brand.split()):
                        if vlm_brand == "Unknown" or fb.lower().startswith(vlm_brand.lower()):
                            clean["brand"] = fb
                    elif vlm_brand == "Unknown":
                        clean["brand"] = fb
            
            # Pack count fallback
            if clean["pack_count"] == 1.0:
                pc = parse_pack_count_from_catalog(catalog_text)
                if pc is not None and pc > 1.0:
                    clean["pack_count"] = float(pc)
            
            # Mass/volume fallback
            if clean["mass"] == 0.0 and clean["volume"] == 0.0:
                val, unit = parse_value_unit_from_catalog(catalog_text)
                if val is not None and unit is not None:
                    m_g, v_ml = convert_to_mass_volume(val, unit)
                    if m_g > 0.0:
                        clean["mass"] = float(m_g)
                    if v_ml > 0.0:
                        clean["volume"] = float(v_ml)
            
            # Keywords merge
            catalog_kw = derive_keywords_from_text(catalog_text)
            merged_kw = list(set(clean["keywords"] + catalog_kw))
            if merged_kw:
                clean["keywords"] = merged_kw
            
            # Category fallback
            cat = categorize_from_text(catalog_text)
            if cat != CATEGORY_FALLBACK and cat != clean["category"]:
                clean["category"] = cat
        
        return clean
    except:
        return default

def process_micro_batch(images, texts, model, processor, tokenizer, device):
    """Process one micro-batch through the model"""
    
    # Trim texts
    trimmed_texts = [t[:MAX_INPUT_CHARS] if isinstance(t, str) else "" for t in texts]
    
    # Build prompts
    prompts = []
    for t in trimmed_texts:
        user_prompt = f'Product Description: "{t}"'
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
        ]
        prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    
    # Process
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = inputs.to(device, non_blocking=True)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    responses = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    del inputs, generated_ids
    
    # Parse responses with catalog fallback
    results = []
    for idx, response in enumerate(responses):
        parsed = parse_vlm_response(response, trimmed_texts[idx])
        results.append(parsed)
    
    return results


# ============================================================================
# Main Processing Loop with Pre-loading Pipeline
# ============================================================================

if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"ERROR: Input file not found at {CSV_FILE_PATH}.")

df = pd.read_csv(CSV_FILE_PATH)
df['image_path'] = df['sample_id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"✅ Loaded {len(df)} records with existing images")

# Start pre-loading pipeline
print("🚀 Starting image pre-loading pipeline...")
preloader = ImagePreloader(
    image_paths=df['image_path'].tolist(),
    texts=df['catalog_content'].tolist(),
    max_workers=NUM_PRELOAD_WORKERS,
    prefetch_batches=IMAGE_PREFETCH_BATCHES,
    batch_size=BATCH_SIZE,
    max_side=MAX_IMAGE_SIDE
)
preloader.start()

all_features = []
processed_total = 0
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 Processing {len(df)} samples with BATCH={BATCH_SIZE}, MICRO_BATCH={MAX_MICRO_BATCH_SIZE}...")
progress = tqdm(total=len(df), desc="VLM extraction", leave=True)

try:
    while True:
        # Get pre-loaded batch (blocking until ready)
        batch_data = preloader.get_batch()
        if batch_data is None:  # End signal
            break
        
        images = batch_data['images']
        texts = batch_data['texts']
        start_idx = batch_data['start_idx']
        end_idx = batch_data['end_idx']
        
        # Process in micro-batches
        micro_bs = MAX_MICRO_BATCH_SIZE
        for micro_start in range(0, len(images), micro_bs):
            micro_end = min(micro_start + micro_bs, len(images))
            
            try:
                results = process_micro_batch(
                    images[micro_start:micro_end],
                    texts[micro_start:micro_end],
                    model, processor, tokenizer, device
                )
                
                # Add sample IDs to results
                for idx, parsed in enumerate(results):
                    parsed['sample_id'] = df['sample_id'].iloc[start_idx + micro_start + idx]
                    all_features.append(parsed)
                
                progress.update(micro_end - micro_start)
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️ OOM at micro_bs={micro_bs}, reducing...")
                    micro_bs = max(1, micro_bs // 2)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        processed_total = end_idx
        
        # Streaming checkpoint - APPEND mode to accumulate all data!
        if processed_total % STREAM_EVERY_N < BATCH_SIZE and len(all_features) > 0:
            tmp_df = pd.DataFrame(all_features)
            for kw in KEYWORD_LIST:
                col = f'flag_{kw.replace(" ", "_")}'
                tmp_df[col] = tmp_df['keywords'].apply(lambda lst: 1 if isinstance(lst, list) and kw in lst else 0)
            tmp_df.drop(columns=['keywords'], inplace=True, errors='ignore')
            final_cols = ['sample_id', 'brand', 'pack_count', 'mass', 'volume', 'category'] + [f'flag_{kw.replace(" ", "_")}' for kw in KEYWORD_LIST]
            tmp_df = tmp_df.reindex(columns=final_cols)
            
            # First checkpoint: write with header, subsequent: append without header
            if processed_total <= STREAM_EVERY_N + BATCH_SIZE:
                tmp_df.to_csv(STREAM_OUT_CSV_FILE, index=False, mode='w')
                total_saved = len(tmp_df)
            else:
                # Check current file size to track total
                if os.path.exists(STREAM_OUT_CSV_FILE):
                    existing = pd.read_csv(STREAM_OUT_CSV_FILE)
                    total_saved = len(existing) + len(tmp_df)
                else:
                    total_saved = len(tmp_df)
                tmp_df.to_csv(STREAM_OUT_CSV_FILE, index=False, mode='a', header=False)
            
            print(f"\n💾 Streamed {len(tmp_df)} new rows (total saved: {total_saved}) at {processed_total}/{len(df)}")
            all_features = []
        
        gc.collect()

finally:
    preloader.stop()
    progress.close()

# Final save
print("\n📊 Assembling final dataset...")

# Process any remaining features (after last streaming checkpoint)
if len(all_features) > 0:
    features_df = pd.DataFrame(all_features)
    # Only process if we have the keywords column
    if 'keywords' in features_df.columns:
        for kw in KEYWORD_LIST:
            col = f'flag_{kw.replace(" ", "_")}'
            features_df[col] = features_df['keywords'].apply(lambda lst: 1 if isinstance(lst, list) and kw in lst else 0)
        features_df.drop(columns=['keywords'], inplace=True, errors='ignore')
    final_cols = ['sample_id', 'brand', 'pack_count', 'mass', 'volume', 'category'] + [f'flag_{kw.replace(" ", "_")}' for kw in KEYWORD_LIST]
    remainder_df = features_df.reindex(columns=final_cols)
else:
    remainder_df = pd.DataFrame()  # Empty, all data in stream

# Load stream data (contains most/all processed samples)
if os.path.exists(STREAM_OUT_CSV_FILE):
    print(f"📂 Loading streamed data from: {STREAM_OUT_CSV_FILE}")
    stream_df = pd.read_csv(STREAM_OUT_CSV_FILE)
    if len(remainder_df) > 0:
        # Combine stream + remainder, remove duplicates
        final_df = pd.concat([stream_df, remainder_df], ignore_index=True).drop_duplicates(subset=['sample_id'], keep='last')
    else:
        # All data is in stream file
        final_df = stream_df
else:
    # No stream file, use remainder only
    final_df = remainder_df

final_df.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"\n🎉 Complete! Saved to: {OUTPUT_CSV_FILE}")
print(f"📊 Processed {len(final_df)} samples")
print(final_df.head())

# ============================================================================
# ML Post-Processing: Brand Gazetteer + Fuzzy Match + Category Classifier
# ============================================================================

print('\n🔧 Starting ML post-processing (brands + categories)...')

try:
    import sys
    import subprocess
    import numpy as np
    from collections import Counter
    
    # Install dependencies if needed
    def _ensure(pkg: str):
        try:
            __import__(pkg.split('==')[0].split('[')[0])
        except:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pkg])
            except:
                pass
    
    _ensure('sentence-transformers')
    _ensure('scikit-learn')
    _ensure('rapidfuzz')
    
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sentence_transformers import SentenceTransformer
    
    # Merge catalog text for ML
    df_merge = final_df.merge(df[['sample_id', 'catalog_content']], on='sample_id', how='left')
    df_merge['catalog_content'] = df_merge['catalog_content'].fillna('')
    
    # --- Brand Gazetteer + Fuzzy Match ---
    def _norm_brand(s: str) -> str:
        s = (s or '').strip()
        s = s.replace("'", "'")
        s = re.sub(r"\s+", ' ', s)
        return s
    
    def _is_brand_suspicious(b: str) -> bool:
        """Minimal check - trust VLM output unless clearly wrong"""
        if not b or b.lower() == 'unknown':
            return True
        bl = b.lower()
        if re.search(r'\b\d+\s*(oz|ounce|ml|g|gram|kg|lb|%)\b', bl):
            return True
        if re.search(r'\b(pack|case)\s*(of|count)', bl):
            return True
        if len(b.split()) > 5:
            return True
        return False
    
    # Build gazetteer from confident brands
    confident_brands = [b for b in final_df['brand'].dropna().map(_norm_brand) if not _is_brand_suspicious(b)]
    brand_freq = Counter(confident_brands)
    gazetteer = sorted([b for b, c in brand_freq.items() if c >= 2 or len(b) <= 12])
    
    # Brand extraction from text (reuse helper from above)
    if len(gazetteer) > 0:
        gazetteer_lower = {b.lower(): b for b in gazetteer}
        gaz_list_lower = list(gazetteer_lower.keys())
        
        def _fix_brand_row(row):
            cur = row['brand']
            txt = row['catalog_content']
            if cur and cur != 'Unknown' and not _is_brand_suspicious(cur):
                return cur
            cand = extract_brand_from_catalog(txt) or 'Unknown'
            if cand != 'Unknown':
                lo = cand.lower()
                if lo in gazetteer_lower:
                    return gazetteer_lower[lo]
                match = rf_process.extractOne(lo, gaz_list_lower, scorer=rf_fuzz.WRatio)
                if match and match[1] >= 90:
                    return gazetteer_lower[match[0]]
                match2 = rf_process.extractOne(lo, gaz_list_lower, scorer=rf_fuzz.token_set_ratio)
                if match2 and match2[1] >= 92:
                    return gazetteer_lower[match2[0]]
            return cur if cur and cur != 'Unknown' else cand
        
        df_merge['brand'] = df_merge.apply(_fix_brand_row, axis=1)
    
    # --- Category Classifier (MiniLM + LogReg) ---
    try:
        device_enc = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device_enc)
        
        # Training data: confident categories (exclude Uncategorized)
        train_mask = (df_merge['category'].fillna('') != CATEGORY_FALLBACK) & (df_merge['catalog_content'].str.len() > 10)
        train_texts = df_merge.loc[train_mask, 'catalog_content'].tolist()
        train_labels = df_merge.loc[train_mask, 'category'].tolist()
        
        if len(train_texts) >= 200:
            print("🔥 Encoding text with SentenceTransformer (GPU)...")
            emb_train = sbert.encode(train_texts, batch_size=2048 if device_enc == 'cuda' else 256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            le = LabelEncoder()
            y = le.fit_transform(train_labels)
            
            # Try GPU-native cuML first (RAPIDS), fall back to scikit-learn
            use_gpu_ml = False
            try:
                import cuml
                import cupy as cp
                print("🚀 Using GPU-native cuML LogisticRegression (no CPU transfer!)...")
                # Convert to GPU arrays (float32 for faster GPU compute)
                X_gpu = cp.asarray(emb_train, dtype=cp.float32)
                y_gpu = cp.asarray(y, dtype=cp.int32)
                # Note: cuML auto-handles multi-class (uses softmax loss), no n_jobs (GPU-native)
                clf = cuml.LogisticRegression(
                    max_iter=1000, 
                    linesearch_max_iter=50,  # Default, but explicit for tuning
                    output_type='cupy',  # Keep outputs on GPU until final transfer
                    verbose=0
                )
                clf.fit(X_gpu, y_gpu)
                use_gpu_ml = True
            except (ImportError, Exception) as e:
                print(f"⚠️ cuML not available ({e}), using CPU scikit-learn...")
                clf = LogisticRegression(max_iter=1000, n_jobs=60, multi_class='multinomial', verbose=0)
                clf.fit(emb_train, y)
            
            # Predict all
            print("🔥 Predicting categories...")
            all_texts = df_merge['catalog_content'].tolist()
            emb_all = sbert.encode(all_texts, batch_size=2048 if device_enc == 'cuda' else 256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            
            if use_gpu_ml:
                # GPU prediction (stays on GPU until final transfer)
                X_all_gpu = cp.asarray(emb_all, dtype=cp.float32)
                proba_gpu = clf.predict_proba(X_all_gpu)  # Returns CuPy array due to output_type='cupy'
                proba = cp.asnumpy(proba_gpu)  # Single transfer: GPU -> CPU for final results only
                del X_all_gpu, proba_gpu  # Free GPU memory
            else:
                # CPU prediction
                proba = clf.predict_proba(emb_all)
            
            pred_idx = proba.argmax(axis=1)
            pred_cat = le.inverse_transform(pred_idx)
            pred_conf = proba.max(axis=1)
            df_merge['category_ml'] = pred_cat
            df_merge['category_ml_conf'] = pred_conf
            
            # Override rules
            def _choose_cat(row):
                cur = row['category'] if isinstance(row['category'], str) else CATEGORY_FALLBACK
                mlc, conf = row['category_ml'], float(row['category_ml_conf'])
                if cur == CATEGORY_FALLBACK and conf >= 0.50:
                    return mlc
                if mlc != cur and conf >= 0.72:
                    return mlc
                return cur
            
            df_merge['category'] = df_merge.apply(_choose_cat, axis=1)
            print(f"✅ Category classification complete ({'GPU-native' if use_gpu_ml else 'CPU'})!")
    except Exception as e:
        print(f'⚠️ Category ML step skipped: {e}')
    
    # Re-emit final columns
    keyword_cols = [c for c in df_merge.columns if c.startswith('flag_')]
    final_cols = ['sample_id', 'brand', 'pack_count', 'mass', 'volume', 'category'] + keyword_cols
    df_final_ml = df_merge.reindex(columns=final_cols)
    
    # Save
    df_final_ml.to_csv(OUTPUT_CSV_FILE, index=False)
    print('✅ Post-processing complete. Updated file saved.')
    print(df_final_ml.head())
    
except Exception as e:
    print(f'⚠️ Skipping ML post-processing: {e}')
    print('Using VLM-only results.')


