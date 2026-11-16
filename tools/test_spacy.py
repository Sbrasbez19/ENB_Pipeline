import spacy
from enbgraph.extraction.stance_parsed import ParsedStanceExtractor
text = ("The PHILIPPINES, with SAUDI ARABIA and CHINA, stressed that technology "
        "transfer was a commitment and opposed linking it to the CDM.")
ex = ParsedStanceExtractor()
doc = ex.nlp(text)
for t in doc:
    print(f"{t.text:12} pos={t.pos_:6} dep={t.dep_:10} head={t.head.text!r} lemma={t.lemma_!r}")
print("lexicon has 'oppose'? ->", ex.lemma_sign.get('oppose'))