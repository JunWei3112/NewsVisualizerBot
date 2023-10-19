from QueryProcessor import PartOfSpeechTagging
from QueryProcessor import DependencyParser
from QueryProcessor import InstructionGenerator

def process_query(query):
    pos_tagged_doc = PartOfSpeechTagging.pos_tagging(query)
    parsed_doc = DependencyParser.parse_dependencies(pos_tagged_doc)
    return InstructionGenerator.generate_instructions(parsed_doc)

if __name__ == "__main__":
    query = 'Place another graph besides the header'
    process_query(query)
