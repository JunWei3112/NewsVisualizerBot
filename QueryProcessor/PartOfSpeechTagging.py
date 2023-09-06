import stanza

def pos_tagging(query):

    # Download and load default processors into a pipeline for English
    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma', download_method=None)

    # Passing text to pipeline instance and annotating the document
    doc = nlp(query)
    return doc

def print_document(doc):
    print('--------POS TAGGING----------')
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f'id: {word.id}\tword: {word.text}\t')
