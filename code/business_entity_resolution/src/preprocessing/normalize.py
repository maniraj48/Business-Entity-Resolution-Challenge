"""
Data normalization module for Business Entity Resolution Challenge.
Provides reusable functions for normalizing business name, address, and country fields.
Maintains open-set country handling and preserves original dataframe fields.
"""

import re
import unicodedata
import pandas as pd
from typing import Optional

# Common business legal suffixes (word-boundary matched)
LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|limited liability company|ltd|limited|corp|corporation|"
    r"co|company|gmbh|pvt|private|services|service|group|solutions|holdings|holding|"
    r"enterprises|enterprise|technologies|technology|tech|sys|systems|system|intl|international|"
    r"sa|pvt ltd|co ltd|llp|plc|nv)\b",
    re.IGNORECASE
)

# Address abbreviations dictionary
ADDRESS_ABBR = {
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "ave.": "avenue", "av": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "ste": "suite", "ste.": "suite",
    "apt": "apartment", "apt.": "apartment",
    "fl": "floor", "fl.": "floor",
    "pkwy": "parkway", "pkwy.": "parkway",
    "hwy": "highway", "hwy.": "highway",
    "ct": "court", "ct.": "court",
    "pl": "place", "pl.": "place",
    "sq": "square", "sq.": "square",
    "bldg": "building", "bldg.": "building",
    "n": "north", "s": "south", "e": "east", "w": "west",
}

# Common country alias mappings (open-set fallback for unknown countries)
COUNTRY_ALIAS = {
    "us": "united states",
    "usa": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "in": "india",
    "ind": "india",
    "fr": "france",
    "fra": "france",
    "de": "germany",
    "deu": "germany",
    "ca": "canada",
    "can": "canada",
    "au": "australia",
    "aus": "australia",
    "jp": "japan",
    "jpn": "japan",
    "cn": "china",
    "chn": "china",
    "br": "brazil",
    "bra": "brazil",
}

def remove_accents(input_str: str) -> str:
    """Normalize unicode characters to ASCII representation."""
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_name(name: Optional[str]) -> str:
    """
    Normalize business name string.
    - Convert to lower case and strip leading/trailing whitespace
    - Remove accents/diacritics
    - Replace '&' with 'and'
    - Remove punctuation except spaces
    - Collapse extra whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    
    text = name.strip().lower()
    text = remove_accents(text)
    
    # Replace ampersands
    text = re.sub(r"&", " and ", text)
    
    # Replace punctuation with single space
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def normalize_name_canonical(name: Optional[str]) -> str:
    """
    Produce a canonical normalized name by stripping legal entity suffixes.
    Useful for strict blocking / indexing key matching.
    """
    norm = normalize_name(name)
    if not norm:
        return ""
    
    # Remove common legal suffixes
    cleaned = LEGAL_SUFFIX_RE.sub("", norm)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else norm

def normalize_address(address: Optional[str]) -> str:
    """
    Normalize business address string.
    - Lowercase and strip accents
    - Expand common address abbreviations
    - Clean punctuation and extra spaces
    """
    if not isinstance(address, str) or not address.strip():
        return ""
    
    text = address.strip().lower()
    text = remove_accents(text)
    
    # Replace punctuation with spaces except hyphens in house numbers/zip codes
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    
    normalized_tokens = []
    for token in tokens:
        clean_tok = token.strip("-.,")
        if clean_tok in ADDRESS_ABBR:
            normalized_tokens.append(ADDRESS_ABBR[clean_tok])
        else:
            normalized_tokens.append(clean_tok)
            
    res = " ".join(normalized_tokens)
    return re.sub(r"\s+", " ", res).strip()

def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country string without imposing a closed set constraint.
    - Lowercase, strip whitespace and accents
    - Standardize common ISO codes / aliases
    - Retain original clean string for unknown/open-set countries
    """
    if not isinstance(country, str) or not country.strip():
        return ""
    
    c_clean = remove_accents(country.strip().lower())
    c_clean = re.sub(r"[^\w\s]", "", c_clean).strip()
    
    return COUNTRY_ALIAS.get(c_clean, c_clean)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe in-place or return copy with normalized columns.
    Preserves original columns ('business_name', 'business_address', 'country').
    Adds 'normalized_business_name', 'canonical_business_name', 
         'normalized_business_address', 'normalized_country'.
    """
    df_out = df.copy()
    
    if "business_name" in df_out.columns:
        df_out["normalized_business_name"] = df_out["business_name"].apply(normalize_name)
        df_out["canonical_business_name"] = df_out["business_name"].apply(normalize_name_canonical)
    else:
        df_out["normalized_business_name"] = ""
        df_out["canonical_business_name"] = ""
        
    if "business_address" in df_out.columns:
        df_out["normalized_business_address"] = df_out["business_address"].apply(normalize_address)
    else:
        df_out["normalized_business_address"] = ""
        
    if "country" in df_out.columns:
        df_out["normalized_country"] = df_out["country"].apply(normalize_country)
    else:
        df_out["normalized_country"] = ""
        
    return df_out
