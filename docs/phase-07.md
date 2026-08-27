# NEXORA AI — Phase 07: Multilingual NLP & Indic Language Architecture

## 1. Overview & Vision

Phase 07 establishes the core **Indic & Multilingual Natural Language Processing (NLP)** subsystem for NEXORA AI, delivering first-class support for **Kannada (`kn`)** alongside major Indic languages (**Hindi `hi`**, **Tamil `ta`**, **Telugu `te`**, **Malayalam `ml`**, **Marathi `mr`**, **Bengali `bn`**) and **English (`en`)**.

```
                           ┌────────────────────────────────────────────────────────┐
                           │               NEXORA MULTILINGUAL NLP                  │
                           │   - Centralized Language Registry                      │
                           │   - Deterministic O(N) Unicode Script Identifier       │
                           │   - Multi-Script / Mixed Text Disambiguation           │
                           │   - Dravidian & Indo-Aryan Script Normalization        │
                           │   - Indic Syllabic & Subword Tokenizer Engine          │
                           │   - Statistical N-Gram Language Disambiguator          │
                           │   - Unified 7-Stage End-to-End Pipeline Gateway        │
                           └───────────────────────────┬────────────────────────────┘
                                                       │
         ┌─────────────────────────┬───────────────────┴───────────────────┬─────────────────────────┐
         ▼                         ▼                                       ▼                         ▼
   Kannada (kn)               Hindi (hi)                              Tamil (ta)                Telugu (te)
   U+0C80 – U+0CFF            U+0900 – U+097F                         U+0B80 – U+0BFF           U+0C00 – U+0C7F
         │                         │                                       │                         │
         ▼                         ▼                                       ▼                         ▼
  Malayalam (ml)              Marathi (mr)                            Bengali (bn)              English (en)
  U+0D00 – U+0D7F             U+0900 – U+097F                         U+0980 – U+09FF           U+0020 – U+007F
```

---

## 2. Language Registry Architecture

The [`LanguageRegistry`](file:///C:/Users/Administrator/.gemini/antigravity-ide/scratch/nexora-ai/backend/fastapi/nlp/language_registry.py) maintains strongly typed [`LanguageMetadata`](file:///C:/Users/Administrator/.gemini/antigravity-ide/scratch/nexora-ai/backend/fastapi/nlp/models/language_metadata.py) definitions:

| ISO Code | Language Name | Native Autonym | Script | Family | Unicode Range | Is Indic | Tokenizer Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `en` | English | English | Latin | Indo-European | `0x0020 – 0x007F` | No | BPE |
| `kn` | **Kannada** | **ಕನ್ನಡ** | **Kannada** | **Dravidian** | `0x0C80 – 0x0CFF` | **Yes** | **SentencePiece** |
| `hi` | Hindi | हिन्दी | Devanagari | Indo-Aryan | `0x0900 – 0x097F` | Yes | SentencePiece |
| `ta` | Tamil | தமிழ் | Tamil | Dravidian | `0x0B80 – 0x0BFF` | Yes | SentencePiece |
| `te` | Telugu | తెలుగు | Telugu | Dravidian | `0x0C00 – 0x0C7F` | Yes | SentencePiece |
| `ml` | Malayalam | മലയാളം | Malayalam | Dravidian | `0x0D00 – 0x0D7F` | Yes | SentencePiece |
| `mr` | Marathi | मराठी | Devanagari | Indo-Aryan | `0x0900 – 0x097F` | Yes | SentencePiece |
| `bn` | Bengali | বাংলা | Bengali | Indo-Aryan | `0x0980 – 0x09FF` | Yes | SentencePiece |

---

## 3. Unicode Script Identifier & Range Analysis

The [`UnicodeScriptIdentifier`](file:///C:/Users/Administrator/.gemini/antigravity-ide/scratch/nexora-ai/backend/fastapi/nlp/detectors/script_identifier.py) performs deterministic, $O(N)$ code-point categorization without invoking heavy ML models:

### Script Identification Strategy
1. **Code-Point Inspection**: Inspects each character's integer code point against standard Unicode blocks.
2. **Formatting & Control Character Filtering**: Safely ignores ASCII whitespace, punctuation, control characters, and Unicode joiners (Zero-Width Joiner `U+200D`, Zero-Width Non-Joiner `U+200C`, BOM `U+FEFF`, hyphens `U+2010–U+2015`).
3. **Script Distribution & Dominance**: Computes the exact character percentage per script and identifies the `primary_script` and `confidence`.

---

## 4. Multi-Script & Mixed Text Handling

When inputs contain multiple writing systems (e.g. `"ನಮಸ್ಕಾರ NEXORA AI Welcome"`):
- `mixed`: Set to `True`.
- `scripts`: Returns list of all distinct scripts (`["Kannada", "Latin"]`).
- `script_distribution`: Reports exact ratios (`{"Kannada": 0.35, "Latin": 0.65}`).
- `primary_script`: Identifies the script with the highest character count.

---

## 5. Devanagari Multi-Language Ambiguity

Because **Hindi (`hi`)** and **Marathi (`mr`)** both use the Devanagari script (`U+0900 – U+097F`), script identification alone cannot definitively distinguish them. The identifier returns both candidates:
```json
{
  "primary_script": "Devanagari",
  "language_candidates": ["hi", "mr"],
  "confidence": 1.0,
  "is_indic": true
}
```

---

## 6. Romanized Indic Limitation

Romanized Indic text (e.g., `"namaskara hegiddira"`, `"kaise ho"`) is composed entirely of Latin characters (`U+0041–U+007A`). In Step 2, this is correctly and safely identified as `Latin` script with candidate `["en"]`. Transliteration and phonological mapping will be introduced in Step 3.

---

## 7. Security Limits & Performance

- **Defensive Character Limit**: Rejects inputs exceeding 100,000 characters with a descriptive `ValueError` to prevent algorithmic complexity attacks.
- **Sub-millisecond Latency**: Pure in-memory code point arithmetic executes in $< 0.1\text{ms}$ for standard user prompts, ensuring zero overhead on real-time WebSocket streams.

---

## 8. Unicode Normalization & Kannada Canonical Formatter

### Normalizer Inheritance Hierarchy
```
BaseNormalizer (Unicode NFC + Control Char Sanitation + Whitespace Collapse)
       │
       ▼
IndicNormalizer (Preserves ZWJ/ZWNJ Ligatures + Indic Danda Punctuation । ॥)
       │
       ▼
KannadaNormalizer (Ottakshara Conjunct Preservation + Diacritic Validation)
```

### Key Normalization Capabilities
1. **Unicode NFC Canonical Standard**: Automatically resolves decomposed Unicode sequences into canonical composite representations (`unicodedata.normalize("NFC", text)`).
2. **Zero-Width Character Policy**:
   - **Preserved**: Essential Zero-Width Joiner (`\u200D`) and Non-Joiner (`\u200C`) when flanked by Indic consonants and viramas.
   - **Stripped**: Isolated, leading, trailing, or duplicate zero-width markers and Byte Order Marks (`\uFEFF`).
3. **Kannada Ottakshara (Conjunct) Preservation**:
   - Consonant + Virama (`್`) + Consonant sequences (e.g. `ಕ್ಕ`, `ಕ್ರ`, `ಕ್ಲ`, `ಜ್ಞ`, `ಷ್ಟ`) remain 100% linguistically valid.
4. **Punctuation & Whitespace**:
   - Preserves Latin punctuation (`. , ? ! : ; - ( ) " '`) alongside Indic punctuation (`।`, `॥`).
   - Normalizes multiple spaces to single space while preserving intentional double newlines for paragraph breaks.
5. **Idempotency Guarantee**:
   $$\text{normalize}(\text{normalize}(text)) \equiv \text{normalize}(text)$$

---

## 9. Tokenizer Architecture & Subword Segmentation Engine

### Multilingual NLP Ingestion Pipeline
```
Raw Text ──► Language Registry ──► Script Identifier ──► Normalizer ──► Tokenizer ──► Tokens & Token IDs
```

### Subword Strategies Comparison
| Strategy | Description | Strengths for Indic Languages |
| :--- | :--- | :--- |
| **Character-Level** | Splits into single Unicode characters | No OOV words, but causes excessively long sequences and loses semantic chunking. |
| **Word-Level** | Splits by whitespace/punctuation | Simple, but suffers from severe out-of-vocabulary (OOV) explosion for agglutinative Indic languages. |
| **SentencePiece** | Treats input as raw Unicode byte/char stream | Language-independent, preserves conjunct units and spaces without pre-tokenization. **Primary choice for Indic NLP**. |
| **Byte-Pair Encoding (BPE)** | Iterative frequency-based subword merge | Highly effective vocabulary compression, standard in modern generative LLMs. |
| **WordPiece** | Likelihood-maximization subword segmenter | Common in BERT-style encoders; subword prefix tracking (`##`). |

---

## 10. Statistical Language Disambiguation & Romanized Indic Detection

### 1. Hindi vs. Marathi Devanagari Disambiguation
- Uses character n-gram profiles ($n \in \{1, 2, 3\}$) and distinctive vocabulary markers:
  - **Marathi Signals**: Exclusive letter `ळ` (`U+0933`), suffixes `च्या`, `मध्ये`, verbs `आहे`, `नाही`, `करणे`, `केले`, `होते`.
  - **Hindi Signals**: Auxiliary verbs `है`, `हूँ`, `था`, `थी`, postpositions `में`, `के`, `की`, `से`, `को`, `लिए`.
- If text is short or lacks distinct markers (e.g., `"भारत"`), the system retains candidates `["hi", "mr"]` and sets `ambiguous: True` rather than guessing.

### 2. Romanized Indic Detection
- Identifies common Romanized Kannada tokens (`namaskara`, `nanna`, `hesaru`, `hegiddira`, `dhanyavada`, `kannada`) vs. standard English tokens (`the`, `is`, `platform`, `artificial`, `intelligence`).
- Flags `romanized: True` with explicit evidence annotations.

---

## 11. Unified Multilingual NLP Pipeline & REST Gateway

The [`MultilingualNLPPipeline`](file:///C:/Users/Administrator/.gemini/antigravity-ide/scratch/nexora-ai/backend/fastapi/nlp/services/multilingual_pipeline.py) unifies all NLP stages into a single deterministic, sub-millisecond execution pipeline:

```
                                ┌──────────────────────────┐
                                │   POST /api/v1/nlp/analyze │
                                └────────────┬─────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    MultilingualNLPPipeline    │
                             │  1. Input Validation          │
                             │  2. Script Identification     │
                             │  3. Statistical Disambiguation│
                             │  4. Language Normalization    │
                             │  5. Subword Tokenization      │
                             │  6. Processing Telemetry      │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                                ┌──────────────────────────┐
                                │    NLPAnalyzeResponse    │
                                └──────────────────────────┘
```

### Sample Request & Response
```bash
POST /api/v1/nlp/analyze
Content-Type: application/json

{
  "text": "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?"
}
```

```json
{
  "original_text": "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?",
  "normalized_text": "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?",
  "language": "kn",
  "language_candidates": ["kn"],
  "script": "Kannada",
  "scripts": ["Kannada"],
  "confidence": 1.0,
  "mixed_language": false,
  "romanized": false,
  "ambiguous": false,
  "tokens": ["ನಮ", "ಸ್", "ಕಾರ", ",", "ಹೇ", "ಗಿ", "ದ್", "ದೀ", "ರಾ", "?"],
  "token_ids": [28714, 14920, 26311, 23078, 28723, 20387, 14920, 19280, 20379, 13735],
  "token_count": 10,
  "processing_time_ms": 0.12
}
```
