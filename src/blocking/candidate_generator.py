"""
Candidate generation / blocking module for Business Entity Resolution Challenge.
Generates candidate entity pairs (S1 -> list of S2/S3 entity IDs) with high candidate recall.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

# Stop words for token blocking
BLOCKING_STOP_WORDS = {
    "and", "the", "inc", "llc", "ltd", "corp", "corporation", "co", "company",
    "gmbh", "pvt", "private", "limited", "services", "service", "group",
    "solutions", "holdings", "holding", "enterprises", "enterprise",
    "technologies", "technology", "tech", "systems", "system", "intl",
    "street", "road", "avenue", "drive", "lane", "suite", "floor", "building",
    "north", "south", "east", "west"
}

class CandidateGenerator:
    def __init__(
        self,
        min_token_len: int = 3,
        max_candidates_per_s1: int = 100,
        enable_fuzzy_fallback: bool = True,
        fuzzy_top_k: int = 15
    ):
        """
        Initialize Candidate Generator with configurable blocking thresholds.
        
        :param min_token_len: Minimum token length to be considered a blocking token.
        :param max_candidates_per_s1: Maximum candidates retained per S1 entity.
        :param enable_fuzzy_fallback: Whether to trigger fuzzy fallback search for entities with low candidate counts.
        :param fuzzy_top_k: Number of candidates to retrieve via fuzzy search if fallback is triggered.
        """
        self.min_token_len = min_token_len
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.enable_fuzzy_fallback = enable_fuzzy_fallback
        self.fuzzy_top_k = fuzzy_top_k
        
    def _extract_tokens(self, text: str) -> List[str]:
        """Extract significant tokens from normalized string."""
        if not text:
            return []
        tokens = text.split()
        return [t for t in tokens if len(t) >= self.min_token_len and t not in BLOCKING_STOP_WORDS]

    def build_index(self, df_candidates: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """
        Build multi-index on S2 and S3 candidate dataset.
        
        Indexes created:
        - exact_name: (normalized_name, country) -> list of entity_ids
        - canonical_name: (canonical_name, country) -> list of entity_ids
        - global_name: normalized_name -> list of entity_ids
        - first_word: (first_word_canonical, country) -> list of entity_ids
        - token_country: (token, country) -> list of entity_ids
        - address_token: (address_token, country) -> list of entity_ids
        """
        index = {
            "exact_name": defaultdict(list),
            "canonical_name": defaultdict(list),
            "global_name": defaultdict(list),
            "first_word": defaultdict(list),
            "token_country": defaultdict(list),
            "address_token": defaultdict(list),
        }
        
        for _, row in df_candidates.iterrows():
            eid = row["entity_id"]
            c_name = row.get("normalized_business_name", "")
            can_name = row.get("canonical_business_name", "")
            addr = row.get("normalized_business_address", "")
            country = row.get("normalized_country", "")
            
            if c_name:
                index["exact_name"][(c_name, country)].append(eid)
                index["global_name"][c_name].append(eid)
                
            if can_name:
                index["canonical_name"][(can_name, country)].append(eid)
                first_w = can_name.split()[0] if can_name.split() else ""
                if len(first_w) >= self.min_token_len:
                    index["first_word"][(first_w, country)].append(eid)
                    
            name_tokens = self._extract_tokens(can_name if can_name else c_name)
            for tok in name_tokens:
                index["token_country"][(tok, country)].append(eid)
                
            addr_tokens = self._extract_tokens(addr)
            for atok in addr_tokens:
                index["address_token"][(atok, country)].append(eid)
                
        return index

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_candidates: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generate candidates for each S1 entity in df_s1 using indexed S2/S3 candidates.
        Returns map: s1_entity_id -> list of candidate entity_ids (S2/S3).
        """
        index = self.build_index(df_candidates)
        
        # Pre-build lookup map for candidates for fast scoring / fuzzy fallback
        cand_map = {}
        for _, row in df_candidates.iterrows():
            cand_map[row["entity_id"]] = {
                "name": row.get("normalized_business_name", ""),
                "address": row.get("normalized_business_address", ""),
                "country": row.get("normalized_country", "")
            }
            
        candidate_results = {}
        
        for _, row in df_s1.iterrows():
            s1_id = row["entity_id"]
            s1_name = row.get("normalized_business_name", "")
            s1_can_name = row.get("canonical_business_name", "")
            s1_addr = row.get("normalized_business_address", "")
            s1_country = row.get("normalized_country", "")
            
            candidates: Set[str] = set()
            
            # 1. Exact normalized name match + country
            if s1_name:
                candidates.update(index["exact_name"].get((s1_name, s1_country), []))
                
            # 2. Canonical name match + country
            if s1_can_name:
                candidates.update(index["canonical_name"].get((s1_can_name, s1_country), []))
                
            # 3. Global normalized name match (across any country)
            if s1_name:
                candidates.update(index["global_name"].get(s1_name, []))
                
            # 4. First word match + country
            if s1_can_name:
                first_w = s1_can_name.split()[0] if s1_can_name.split() else ""
                if len(first_w) >= self.min_token_len:
                    candidates.update(index["first_word"].get((first_w, s1_country), []))
                    
            # 5. Token overlap + country
            name_tokens = self._extract_tokens(s1_can_name if s1_can_name else s1_name)
            for tok in name_tokens:
                candidates.update(index["token_country"].get((tok, s1_country), []))
                
            # 6. Address token overlap + country
            addr_tokens = self._extract_tokens(s1_addr)
            for atok in addr_tokens:
                candidates.update(index["address_token"].get((atok, s1_country), []))
                
            # 7. Fuzzy fallback if 0 candidates found and fuzzy fallback enabled
            if len(candidates) == 0 and self.enable_fuzzy_fallback and s1_name:
                # Find best matches in same country candidates
                same_country_cands = [
                    cid for cid, cinfo in cand_map.items() 
                    if not s1_country or cinfo["country"] == s1_country
                ]
                if not same_country_cands:
                    same_country_cands = list(cand_map.keys())
                    
                if same_country_cands:
                    cand_names = [cand_map[cid]["name"] for cid in same_country_cands]
                    matches = process.extract(
                        s1_name,
                        cand_names,
                        scorer=fuzz.token_set_ratio,
                        limit=self.fuzzy_top_k
                    )
                    for match in matches:
                        cand_idx = match[2]
                        if match[1] >= 50:  # Fuzzy threshold score
                            candidates.add(same_country_cands[cand_idx])
                            
            # Prune candidates if count exceeds max_candidates_per_s1
            candidate_list = list(candidates)
            if len(candidate_list) > self.max_candidates_per_s1 and s1_name:
                # Rank candidate set by fast string similarity to keep top N
                scored = []
                for cid in candidate_list:
                    cname = cand_map[cid]["name"]
                    score = fuzz.token_set_ratio(s1_name, cname)
                    scored.append((score, cid))
                scored.sort(key=lambda x: x[0], reverse=True)
                candidate_list = [cid for _, cid in scored[:self.max_candidates_per_s1]]
                
            candidate_results[s1_id] = candidate_list
            
        return candidate_results
