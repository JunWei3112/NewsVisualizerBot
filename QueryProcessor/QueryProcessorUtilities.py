# Universal POS (part-of-speech) tags
UPOS_ADJECTIVE = 'ADJ'
UPOS_ADPOSITION = 'ADP'
UPOS_ADVERB = 'ADV'
UPOS_AUXILIARY = 'AUX'
UPOS_COORDINATING_CONJUNCTION = 'CCONJ'
UPOS_DETERMINER = 'DET'
UPOS_INTERJECTION = 'INTJ'
UPOS_NOUN = 'NOUN'
UPOS_NUMERAL = 'NUM'
UPOS_PARTICLE = 'PART'
UPOS_PRONOUN = 'PRON'
UPOS_PROPER_NOUN = 'PROPN'
UPOS_PUNCTUATION = 'PUNCT'
UPOS_SUBORDINATING_CONJUNCTION = 'SCONJ'
UPOS_SYMBOL = 'SYM'
UPOS_VERB = 'VERB'
UPOS_OTHER = 'X'

# Dependency Relations: Core dependents of clausal predicates (Nominal dep)
DEPREL_NOMINAL_SUBJECT = 'nsubj'
DEPREL_PASSIVE_NOMINAL_SUBJECT = 'nsubj:pass'
DEPREL_OUTER_CLAUSE_NOMINAL_SUBJECT = 'nsubj:outer'
DEPREL_DIRECT_OBJECT = 'obj'
DEPREL_INDIRECT_OBJECT = 'iobj'

# Dependency Relations: Core dependents of clausal predicates (Predicate dep)
DEPREL_CLAUSAL_SUBJECT = 'csubj'
DEPREL_CLAUSAL_PASSIVE_SUBJECT = 'csubj:pass'
DEPREL_OUTER_CLAUSE_CLAUSAL_SUBJECT = 'csubj:outer'
DEPREL_CLAUSAL_COMPLEMENT = 'ccomp'
DEPREL_OPEN_CLAUSAL_COMPLEMENT = 'xcomp'

# Dependency Relations: Non-core dependents of clausal predicates (Nominal dep)
DEPREL_OBLIQUE_NOMINAL = 'obl'
DEPREL_OBL_NOUN_PHRASE_AS_ADVERBIAL_MODIFIER = 'obl:npmod'
DEPREL_OBL_TEMPORAL_MODIFIER = 'obl:tmod'

# Dependency Relations: Non-core dependents of clausal predicates (Predicate dep)
DEPREL_ADVERBIAL_CLAUSE_MODIFIER = 'advcl'
DEPREL_ADVERBIAL_RELATIVE_CLAUSE_MODIFIER = 'advcl:relcl'

# Dependency Relations: Non-core dependents of clausal predicates (Modifier word)
DEPREL_ADVERBIAL_MODIFIER = 'advmod'

# Dependency Relations: Special Clausal Dependents (Nominal dep)
DEPREL_VOCATIVE = 'vocative'
DEPREL_DISCOURSE_ELEMENT = 'discourse'
DEPREL_EXPLETIVE = 'expl'

# Dependency Relations: Special Clausal Dependents (Auxiliary)
DEPREL_AUXILIARY = 'aux'
DEPREL_PASSIVE_AUXILIARY = 'aux:pass'
DEPREL_COPULA = 'cop'

# Dependency Relations: Special Clausal Dependents (Other)
DEPREL_MARKER = 'mark'

# Dependency Relations: Noun Dependents (Nominal dep)
DEPREL_NUMERIC_MODIFIER = 'nummod'
DEPREL_APPOSITIONAL_MODIFIER = 'appos'
DEPREL_NOMINAL_MODIFIER = 'nmod'
DEPREL_NMOD_NOUN_PHRASE_AS_ADVERBIAL_MODIFIER = 'nmod:npmod'
DEPREL_NMOD_TEMPORAL_MODIFIER = 'nmod:tmod'
DEPREL_NMOD_POSSESSIVE_NOMINAL_MODIFIER = 'nmod:poss'

# Dependency Relations: Noun Dependents (Predicate dep)
DEPREL_CLAUSAL_MODIFIER_OF_NOUN = 'acl'
DEPREL_ADNOMINAL_RELATIVE_CLAUSE_MODIFIER = 'acl:relcl'

# Dependency Relations: Noun Dependents (Modifier word)
DEPREL_ADJECTIVAL_MODIFIER = 'amod'
DEPREL_DETERMINER = 'det'
DEPREL_PREDETERMINER = 'det:predet'

# Dependency Relations: Compounding and unanalyzed
DEPREL_COMPOUND = 'compound'
DEPREL_PHRASAL_VERB_PARTICLE = 'compound:prt'
DEPREL_FIXED_MULTIWORD_EXPRESSION = 'fixed'
DEPREL_FLAT = 'flat'
DEPREL_FOREIGN_WORDS = 'flat:foreign'
DEPREL_GOES_WITH = 'goeswith'

# Dependency Relations: Coordination
DEPREL_CONJUNCT = 'conj'
DEPREL_COORDINATION = 'cc'
DEPREL_PRECONJUNCT = 'cc:preconj'

# Dependency Relations: Case-marking, prepositions, possessive
DEPREL_CASE_MARKING = 'case'

# Dependency Relations: Loose joining relations
DEPREL_LIST = 'list'
DEPREL_DISLOCATED_ELEMENTS = 'dislocated'
DEPREL_PARATAXIS = 'parataxis'
DEPREL_ORPHAN_TO_ORPHAN_RELATION_IN_GAPPING = 'orphan'
DEPREL_OVERRIDDEN_DISFLUENCY = 'reparandum'

# Dependency Relations: Other (Sentence head)
DEPREL_ROOT = 'root'

# Dependency Relations: Other (Punctuation)
DEPREL_PUNCTUATION = 'punct'

# Dependency Relations: Other (Unspecified dependency)
DEPREL_DEPENDENT = 'dep'

def check_for_upos_type(word, upos):
    return word.upos == upos

def check_for_dependency_relation(word, deprel):
    return word.deprel == deprel

def combine_two_words(word1, word2):
    if word1.head == word2.id:
        word2.text = word1.text + ' ' + word2.text
        return word2
    else:
        word1.text = word1.text + ' ' + word2.text
        return word1
