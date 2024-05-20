import spacy
import spacy.displacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
import json
import fr_core_news_md

nlp = spacy.load("model-best")
doc = nlp("['N°' 'Designiation' 'Unite' 'QT' 'Prix Unitaire' 'Montant ht']")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)