from QueryProcessor.QueryProcessorUtilities import *

def generate_instructions(parsed_doc):
    verb_and_noun_pairs = extract_all_verb_and_noun_pairs(parsed_doc)
    instructions = ""
    for pair in verb_and_noun_pairs:
        instructions += '----------------------\n'
        instructions += f'VERB:  id: {pair[0].id}\tword: {pair[0].text}\tupos: {pair[0].upos}\tdeprel: {pair[0].deprel}\n'
        instructions += f'NOUN:  id: {pair[1].id}\tword: {pair[1].text}\tupos: {pair[1].upos}\tdeprel: {pair[1].deprel}\n'
        instructions += '----------------------\n'
        # print('----------------------')
        # print(f'VERB:  id: {pair[0].id}\tword: {pair[0].text}\tupos: {pair[0].upos}\tdeprel: {pair[0].deprel}')
        # print(f'NOUN:  id: {pair[1].id}\tword: {pair[1].text}\tupos: {pair[1].upos}\tdeprel: {pair[1].deprel}')
        # print('----------------------')
    return instructions

# Extract all word pairs (word1, word2) where word1 is a verb and is a parent of word2 (which is a noun) in the dependency relation tree.
def extract_all_verb_and_noun_pairs(doc):
    verb_and_noun_pairs = []
    for sentence in doc.sentences:
        for word in sentence.words:
            parent_word = sentence.words[word.head - 1]
            if check_for_upos_type(word, UPOS_NOUN) and parent_word.id > 0 and check_for_upos_type(parent_word, UPOS_VERB):
                verb_and_noun_pair = [parent_word, word]
                verb_and_noun_pairs.append(verb_and_noun_pair)
    return verb_and_noun_pairs


