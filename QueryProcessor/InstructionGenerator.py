from QueryProcessor.QueryProcessorUtilities import *

def generate_instructions(parsed_doc):
    verb_and_noun_pairs = extract_all_verb_and_noun_pairs(parsed_doc)
    instructions = ""
    add_targets = []
    delete_targets = []
    edit_targets = []
    for pair in verb_and_noun_pairs:
        verb = pair[0]
        noun = pair[1]
        if noun.deprel == DEPREL_DIRECT_OBJECT:
            if verb.text.lower() in ADD_SYNONYMS:
                add_targets.append(noun)
            elif verb.text.lower() in DELETE_SYNONYMS:
                delete_targets.append(noun)
            elif verb.text.lower() in EDIT_SYNONYMS:
                edit_targets.append(noun)
            else:
                instructions += 'UNRECOGNISED COMMAND!'

        current_set_of_instructions = '----------------------\n'
        current_set_of_instructions += f'VERB:  id: {pair[0].id}\tword: {pair[0].text}\tupos: {pair[0].upos}\tdeprel: {pair[0].deprel}\n'
        current_set_of_instructions += f'NOUN:  id: {pair[1].id}\tword: {pair[1].text}\tupos: {pair[1].upos}\tdeprel: {pair[1].deprel}\n'
        current_set_of_instructions += '----------------------\n'
        print(current_set_of_instructions)
    instructions += format_instructions(add_targets, delete_targets, edit_targets)
    print('--------FINAL INSTRUCTIONS---------')
    print(instructions)
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

def format_instructions(add_targets, delete_targets, edit_targets):
    instructions = ''
    for add_target in add_targets:
        instructions += '[ADD] ' + add_target.text + '\n'
    for delete_target in delete_targets:
        instructions += '[DELETE] ' + delete_target.text + '\n'
    for edit_target in edit_targets:
        instructions += '[EDIT] ' + edit_target.text + '\n'
    return instructions
