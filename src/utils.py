import re
from nltk.stem.isri import ISRIStemmer

# Regex patterns
elong_re = re.compile(r"([\u0621-\u064A])\1{2,}")
ar_token_re = re.compile(r"[\u0621-\u064A]+")
ar_diacritics_re = re.compile(r"[\u064B-\u0652\u0670]")
tatweel_re = re.compile(r"\u0640")
non_arabic_re = re.compile(r"[^\u0600-\u06FF\s]")

# Lists
verb_prefixes = ("ي", "ت", "ن", "أ", "س")
verb_suffixes = ("ون", "ان", "ين", "وا", "تم", "نا", "ت", "ة")

not_verb_words = {
    "أطفال", "أسبوع", "أستاذ", "أمام", "أماني", "أدب", "أثر",
    "أعمال", "أحداث", "أهداف", "أدوات", "أمور", "أشخاص",
    "تحليل", "تطوير", "تصميم", "تقنية", "نتائج", "نظام",
    "يوم", "يمين", "يسار", "نموذج", "نظرية"
}

not_verb_suffixes = ["ات", "يات", "ية", "ونه", "ونها", "ياته", "ياتها"]

not_dual_words = {
    "كان", "زمان", "مكان", "لسان", "إنسان", "بنيان", "ميزان",
    "يمين", "شعبان", "رمضان", "عثمان", "سليمان", "لبنان", "عمان"
}

stopwords = {
    "في", "من", "على", "إلى", "عن", "أن", "إن", "كان", "كانت",
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم", "هن",
    "كما", "لكن", "بل", "أو", "و", "ثم", "حيث", "لقد",
    "قد", "ما", "لا", "لم", "لن", "إذ", "إذا", "إما",
    "ايضا", "أيضاً", "أيضا", "مع", "بين", "أحد",
    "أمام", "عند", "حتى", "الذي", "التي", "اللذان",
    "اللتان", "الذين", "اللاتي", "اللواتي"
}

stemmer = ISRIStemmer()

# Functions
def normalize_ar(text):
    s = str(text)
    s = ar_diacritics_re.sub("", s)
    s = tatweel_re.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي")
    s = s.replace("ة", "ه")
    return s

def tokens_ar(text):
    return ar_token_re.findall(normalize_ar(text))

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords]

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

def is_verb(token):
    if len(token) < 3 or token.startswith("ال") or token in not_verb_words:
        return False
    if any(token.endswith(suf) for suf in not_verb_suffixes):
        return False
    return token[0] in verb_prefixes or any(token.endswith(s) for s in verb_suffixes)

def is_dual(token):
    if len(token) < 4 or token in not_dual_words:
        return False
    if not (token.endswith("ان") or token.endswith("ين")):
        return False
    before = token[-3]
    if before in "اوي":
        return False
    if token.startswith("ال"):
        base = token[2:]
        if len(base) < 4:
            return False
    return True