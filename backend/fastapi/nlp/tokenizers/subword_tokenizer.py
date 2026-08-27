import re
import unicodedata
from typing import Dict, List, Optional
from nlp.tokenizers.base_tokenizer import (
    BaseTokenizer,
    TokenizationResult,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SPECIAL_TOKENS_MAP,
    ID_TO_SPECIAL_TOKENS,
    MAX_TOKEN_COUNT,
)
from nlp.normalizers.base_normalizer import base_normalizer
from nlp.normalizers.indic_normalizer import indic_normalizer
from nlp.normalizers.kannada_normalizer import kannada_normalizer

# Regular expression that matches Indic conjunct units (Consonant + Virama + Consonant...)
# or Consonant + Matras, preserving whole syllabic graphemes together.
# Indic viramas: U+094D (Devanagari), U+09CD (Bengali), U+0BCD (Tamil), U+0C4D (Telugu), U+0C8D/U+0CCD (Kannada), U+0D4D (Malayalam)
INDIC_GRAPHEME_PATTERN = re.compile(
    r"(?:[\u0900-\u0D7F][\u094D\u09CD\u0BCD\u0C4D\u0CCD\u0D4D]\u200D?[\u0900-\u0D7F]|[\u0900-\u0D7F][\u0900-\u0D7F\u0CBE-\u0CCC\u093E-\u094C\u0BBE-\u0BCC\u0C3E-\u0C4C\u0D3E-\u0D4C]*|[a-zA-Z0-9]+|[^\s\w])",
    re.UNICODE,
)


class SubwordTokenizer(BaseTokenizer):
    """
    Multilingual Subword & Syllabic Tokenizer for NEXORA AI.
    Integrates with the normalizer pipeline, preserves Kannada Ottakshara conjuncts,
    and provides deterministic encoding/decoding for testing and development.
    
    NOTE: Development tokenizer IDs are generated deterministically for testing
    and are not compatible with external pretrained LLM vocabularies.
    """

    def __init__(
        self,
        name: str = "nexora-indic-subword",
        language: str = "kn",
        vocabulary_size: int = 32000,
    ):
        super().__init__(name=name, language=language, vocabulary_size=vocabulary_size)
        # Select appropriate normalizer
        if self.language == "kn":
            self.normalizer = kannada_normalizer
        elif self.language in ("hi", "mr", "ta", "te", "ml", "bn"):
            self.normalizer = indic_normalizer
        else:
            self.normalizer = base_normalizer

        # In-memory dynamic token <-> ID registry for reversible decode during session
        self._token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS_MAP)
        self._id_to_token: Dict[int, str] = dict(ID_TO_SPECIAL_TOKENS)

    def _get_or_create_id(self, token: str) -> int:
        """Returns deterministic token ID, reserving IDs 0-9 for special tokens."""
        if token in self._token_to_id:
            return self._token_to_id[token]

        # Generate positive deterministic hash within vocab bounds
        # Use abs(hash) % (vocab_size - 100) + 100 to prevent collisions with special tokens
        token_hash = abs(hash(token)) % (self.vocabulary_size - 100) + 100
        
        # In case of hash collision with an existing different token in session cache:
        candidate_id = token_hash
        while candidate_id in self._id_to_token and self._id_to_token[candidate_id] != token:
            candidate_id = (candidate_id + 1) % self.vocabulary_size
            if candidate_id < 100:
                candidate_id = 100

        self._token_to_id[token] = candidate_id
        self._id_to_token[candidate_id] = token
        return candidate_id

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[str]:
        """
        Normalizes input and segments text into graphemic and subword token strings,
        strictly preserving Kannada Ottaksharas and Indic composite ligatures.
        """
        if not text or not isinstance(text, str):
            return []

        self.validate_input(text)

        # 1. Normalize text through language-aware normalizer
        normalized = self.normalizer.normalize(text)
        if not normalized:
            return []

        # 2. Extract tokens matching words, Indic conjunct units, numbers, and punctuation
        tokens: List[str] = []
        
        # Match words or sequences of characters
        words = normalized.split()
        for i, word in enumerate(words):
            # Split word into grapheme clusters / punctuation units
            word_tokens = INDIC_GRAPHEME_PATTERN.findall(word)
            if word_tokens:
                # Mark first token with whitespace indicator if SentencePiece-style
                tokens.extend(word_tokens)
            else:
                tokens.append(word)

        if add_special_tokens:
            tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]

        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text into a sequence of integer token IDs."""
        tokens = self.tokenize(text, add_special_tokens=add_special_tokens)
        token_ids: List[int] = []
        for t in tokens:
            token_ids.append(self._get_or_create_id(t))
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into text.
        Handles punctuation attachment and reconstructs word boundaries.
        """
        if not token_ids:
            return ""

        self.validate_token_ids(token_ids)

        token_strings: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in ID_TO_SPECIAL_TOKENS:
                continue
            token_str = self._id_to_token.get(tid, UNK_TOKEN)
            token_strings.append(token_str)

        if not token_strings:
            return ""

        # Reconstruct text with spacing heuristics for words vs punctuation
        result = []
        for i, tok in enumerate(token_strings):
            if i == 0:
                result.append(tok)
            elif re.match(r"^[\.,\?!:;\-\(\)\"\'।॥]+$", tok):
                # Attach punctuation directly to previous token
                result.append(tok)
            elif len(tok) == 1 and unicodedata.category(tok).startswith("M"):
                # Combining matra attached to previous character
                result.append(tok)
            else:
                result.append(" " + tok)

        return "".join(result).strip()

    def tokenize_with_result(
        self, text: str, add_special_tokens: bool = False
    ) -> TokenizationResult:
        """Executes tokenization and packages output in a strongly typed TokenizationResult."""
        raw_text = text if isinstance(text, str) else ""
        normalized = self.normalizer.normalize(raw_text) if raw_text else ""
        tokens = self.tokenize(raw_text, add_special_tokens=add_special_tokens)
        token_ids = [self._get_or_create_id(t) for t in tokens]

        special_count = sum(1 for t in tokens if t in SPECIAL_TOKENS_MAP)
        unk_count = sum(1 for t in tokens if t == UNK_TOKEN)

        return TokenizationResult(
            original_text=raw_text,
            normalized_text=normalized,
            tokens=tokens,
            token_ids=token_ids,
            token_count=len(tokens),
            language=self.language,
            tokenizer_type="development_fallback",
            unknown_token_count=unk_count,
            special_tokens_count=special_count,
        )
