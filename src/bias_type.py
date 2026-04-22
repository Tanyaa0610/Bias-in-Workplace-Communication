def detect_bias_type(text, sentiment, politeness, semantic=0):

    t = text.lower()

    if semantic > 0.2:
        return "Stereotype Bias"

    if "do this" in t or "submit now" in t:
        return "Authority Bias"

    if sentiment < -0.3:
        return "Tone Bias"

    return "Neutral"