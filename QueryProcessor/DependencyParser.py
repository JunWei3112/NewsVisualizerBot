import stanza
from stanza.models.common.doc import Document

def parse_dependencies(pos_tagged_doc):

    # Download and load default processors into a pipeline for English
    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='depparse', download_method=None, depparse_pretagged=True)

    doc = nlp(pos_tagged_doc)
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
    current_word_index = 1
    for sentence in doc.sentences:
        current_word_list = []
        sentence_length = len(sentence.words)
        for index, word in enumerate(sentence.words):
            if (index + 1) < sentence_length and word.upos == 'NOUN' and sentence.words[index + 1].upos == 'NOUN':
                combined_nouns = combine_two_nouns(word, sentence.words[index + 1])
                sentence.words = correct_head_indexes_in_sentence(index, index + 1, sentence.words)
                sentence.words[index + 1] = combined_nouns
            else:
                sentence.words = correct_head_indexes_in_sentence(word.id, current_word_index, sentence.words)
                current_word_list = correct_head_indexes_in_list(word.id, current_word_index, current_word_list)
                word.id = current_word_index
                current_word_index += 1
                current_word_list.append(word.to_dict())
        new_doc_list.append(current_word_list)
    new_doc = Document(new_doc_list)
    return new_doc

def combine_two_nouns(word1, word2):
    combined = word2
    combined.text = word1.text + ' ' + word2.text
    return combined

def correct_head_indexes_in_sentence(old_id, new_id, words):
    for word in words:
        if word.head == old_id:
            word.head = new_id
    return words

def correct_head_indexes_in_list(old_id, new_id, word_list):
    for index, word in enumerate(word_list):
        if word["head"] == old_id:
            word["head"] = new_id
        word_list[index] = word
    return word_list
