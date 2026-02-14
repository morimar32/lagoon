"""Minimal English stop words for filtering noise from unknown word lists."""

STOP_WORDS: frozenset[str] = frozenset({
    # Single letters (from contractions, standalone, enumeration)
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Determiners and articles
    "the", "an",
    # Conjunctions and prepositions
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of",
    "with", "by", "as", "into", "from", "about", "between", "through",
    "during", "before", "after", "above", "below", "under", "over",
    "until", "against",
    # Pronouns
    "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "his", "its", "our", "your", "their",
    "mine", "yours", "hers", "ours", "theirs",
    "myself", "himself", "herself", "itself", "ourselves", "themselves",
    "yourself", "yourselves",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose",
    # Be/have/do forms
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    # Modals
    "will", "would", "shall", "should", "may", "might", "must",
    "can", "could",
    # Adverbs and other function words
    "not", "no", "nor", "so", "too", "very", "just",
    "how", "when", "where", "why", "than", "then",
    "now", "here", "there", "also", "only", "still",
    "well", "back", "even", "once",
    "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "any", "own", "same",
    "up", "down", "out", "off",
    # Contraction fragments (after [a-z]+ regex splitting)
    "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
    "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
    "ll", "ve", "re",
})
