import stanza
from stanza.models.common.doc import Document
from QueryProcessor.QueryProcessorUtilities import *

def parse_dependencies(pos_tagged_doc):

    # Download and load default processors into a pipeline for English
    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='depparse', download_method=None, depparse_pretagged=True)

    doc = nlp(pos_tagged_doc)
    print_document(doc)
    print('----------------')
    doc = combine_consecutive_nouns(doc)
    print_document(doc)
    return doc

def print_document(doc):
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f'id: {word.id}\tword: {word.text}\thead: {sentence.words[word.head - 1].text if word.head > 0 else "root"}\tupos: {word.upos}\tdeprel: {word.deprel}')

# TODO: May need to combine consecutive nouns and adjectives too
def combine_consecutive_nouns(doc):
    new_doc_list = []
    for sentence in doc.sentences:
        current_word_index = 1
        current_word_list = []
        for index, word in enumerate(sentence.words):
            if check_for_noun_compound(sentence, index) or check_for_verb_compound(sentence, index):
                compound_word = combine_two_words(word, sentence.words[index + 1])
                sentence.words = correct_head_indexes_in_sentence(index, index + 1, sentence.words)
                sentence.words[index + 1] = compound_word
            else:
                sentence.words = correct_head_indexes_in_sentence(word.id, current_word_index, sentence.words)
                current_word_list = correct_head_indexes_in_list(word.id, current_word_index, current_word_list)
                word.id = current_word_index
                current_word_index += 1
                current_word_list.append(word.to_dict())
        new_doc_list.append(current_word_list)
    new_doc = Document(new_doc_list)
    return new_doc

# A noun compound occurs when neighboring words are of upos 'NOUN' and the preceding word has deprel 'compound'
def check_for_noun_compound(sentence, current_word_index):
    if (current_word_index + 1) < len(sentence.words):
        current_word = sentence.words[current_word_index]
        next_word = sentence.words[current_word_index + 1]
        return check_for_upos_type(current_word, UPOS_NOUN) and \
            check_for_upos_type(next_word, UPOS_NOUN) and \
            check_for_dependency_relation(current_word, DEPREL_COMPOUND)
    else:
        return False

def check_for_verb_compound(sentence, current_word_index):
    if (current_word_index + 1) < len(sentence.words):
        current_word = sentence.words[current_word_index]
        next_word = sentence.words[current_word_index + 1]
        return check_for_upos_type(current_word, UPOS_VERB) and \
            check_for_upos_type(next_word, UPOS_ADPOSITION) and \
            check_for_dependency_relation(next_word, DEPREL_PHRASAL_VERB_PARTICLE)
    else:
        return False

def correct_head_indexes_in_sentence(old_id, new_id, words):
    for word in words:
        if word.head == old_id + 1:
            word.head = new_id + 1
    return words

def correct_head_indexes_in_list(old_id, new_id, word_list):
    for index, word in enumerate(word_list):
        if word["head"] == old_id + 1:
            word["head"] = new_id + 1
        word_list[index] = word
    return word_list
