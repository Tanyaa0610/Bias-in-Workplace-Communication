import re


def counterfactual(text):
    """
    Convert biased sentence into neutral sentence using:
    1. Gender neutralization
    2. Stereotype word replacement
    3. Authority tone softening
    4. Diminutive/undermining phrase removal
    """

    if not isinstance(text, str) or not text.strip():
        return ""

    t = text.lower()

    # --------------------------------------------------------
    # 1. GENDER NEUTRALIZATION
    # --------------------------------------------------------
    gender_map = {
        r"\bhe\b":       "they",
        r"\bshe\b":      "they",
        r"\bhim\b":      "them",
        r"\bher\b":      "them",
        r"\bhis\b":      "their",
        r"\bhers\b":     "theirs",
        r"\bhimself\b":  "themselves",
        r"\bherself\b":  "themselves",
        r"\bman\b":      "person",
        r"\bwoman\b":    "person",
        r"\bboy\b":      "individual",
        r"\bgirl\b":     "individual",
        r"\bgentleman\b": "person",
        r"\blady\b":     "person",
    }
    for pattern, replacement in gender_map.items():
        t = re.sub(pattern, replacement, t)

    # --------------------------------------------------------
    # 2. STEREOTYPE WORD REPLACEMENT
    # --------------------------------------------------------
    stereotype_map = {
        # emotion-based
        "emotional":   "expressive",
        "hysterical":  "strongly concerned",
        "irrational":  "reacting under pressure",
        "dramatic":    "expressive",
        "sensitive":   "perceptive",
        # weakness-based
        "weak":        "may require additional support",
        "fragile":     "requires a supportive environment",
        "delicate":    "thoughtful",
        "timid":       "reserved",
        "passive":     "considered in approach",
        # authority/aggression
        "bossy":       "directive",
        "aggressive":  "assertive",
        "pushy":       "persistent",
        "difficult":   "has a distinct working style",
        "demanding":   "sets high standards",
        "controlling": "detail-oriented",
        "dominant":    "takes initiative",
        "manipulative":"strategic",
        "catty":       "candid",
        "bitchy":      "direct",
        "shrill":      "emphatic",
    }
    for word, replacement in stereotype_map.items():
        t = re.sub(rf"\b{re.escape(word)}\b", replacement, t)

    # --------------------------------------------------------
    # 3. UNDERMINING / DIMINUTIVE PHRASES
    # --------------------------------------------------------
    undermining_map = {
        "not technical enough":   "developing technical expertise",
        "not analytical enough":  "growing analytical skills",
        "can't handle":           "is working through",
        "cannot handle":          "is working through",
        "too soft":               "empathetic in approach",
        "too emotional":          "expressive",
        "just a":                 "a",
        "only a":                 "a",
        "merely a":               "a",
    }
    for phrase, replacement in undermining_map.items():
        t = t.replace(phrase, replacement)

    # --------------------------------------------------------
    # 4. AUTHORITY → POLITE TONE
    # --------------------------------------------------------
    authority_map = {
        "do this immediately":      "please prioritize this",
        "do this now":              "please do this when possible",
        "do this":                  "please do this",
        "submit now":               "please submit at your earliest convenience",
        "submit immediately":       "please submit when ready",
        "complete immediately":     "please complete this when possible",
        "send this now":            "please send this",
        "send this":                "please send this",
        "finish this now":          "please finish this when possible",
        "finish this":              "please finish this",
        "you must":                 "it would be helpful if you could",
        "you need to":              "it would be appreciated if you could",
        "you have to":              "please consider",
    }
    for phrase, replacement in authority_map.items():
        t = t.replace(phrase, replacement)

    # --------------------------------------------------------
    # 5. CLEAN UP SPACING
    # --------------------------------------------------------
    t = re.sub(r"\s+", " ", t).strip()

    # --------------------------------------------------------
    # 6. CAPITALIZE FIRST LETTER
    # --------------------------------------------------------
    if t:
        t = t[0].upper() + t[1:]

    return t


def batch_mitigate(texts):
    """
    Apply counterfactual mitigation to a list of texts.
    Returns list of (original, mitigated) tuples.
    """
    return [(text, counterfactual(text)) for text in texts]


def mitigation_report(texts, predictions):
    """
    Given raw texts and binary predictions (1=biased),
    returns a DataFrame with original, mitigated, and change flag.
    """
    import pandas as pd

    rows = []
    for text, pred in zip(texts, predictions):
        mitigated = counterfactual(text) if pred == 1 else text
        rows.append({
            "original":  text,
            "mitigated": mitigated,
            "flagged":   bool(pred),
            "changed":   text.lower().strip() != mitigated.lower().strip()
        })
    return pd.DataFrame(rows)