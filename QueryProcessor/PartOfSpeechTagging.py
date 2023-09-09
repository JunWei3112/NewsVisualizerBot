import stanza
from QueryProcessor.QueryProcessorUtilities import *

def pos_tagging(query):

    # Download and load default processors into a pipeline for English
    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma', download_method=None)

    # Passing text to pipeline instance and annotating the document
    doc = nlp(query)
    filter_stop_nouns(doc)
    return doc

def print_document(doc):
    print('--------POS TAGGING----------')
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f'id: {word.id}\tword: {word.text}\t')

def filter_stop_nouns(doc):
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() == 'right' or word.text.lower() == 'left':
                word.upos = UPOS_ADJECTIVE
