import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

def load_knowledge(path_to_file):
    knowledge = []
    with open(path_to_file, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                knowledge.append(json_object['text'])
            except json.JSONDecodeError:
                print(f"Ignoring invalid JSON object: {line}")
    return ' '.join(knowledge)

def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=10, min_length=1, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

def main():
    instruction = f'Instruction: You are a chatbot and talk to other people like a regular person'
    knowledge = ''
    
    brain_path = 'brain.json'
    if os.path.exists(brain_path):
        knowledge = load_knowledge(brain_path)
        print("Prior knowledge loaded from brain.json.")
    else:
        print("No prior knowledge found. Continuing without it.")

    while True:    
        dialog = input("You: ")
        response = generate(instruction, knowledge, dialog)
        print("Bot :", response)

if __name__ == "__main__":
    main()
