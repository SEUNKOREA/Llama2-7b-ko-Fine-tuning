import json
from data_utils import cleanhtml
import datasets as ds

if __name__ == '__main__':
    ### Load raw data
    data_path = './newfile.json' # KorQuAD2.0 https://github.com/korquad/korquad.github.io/tree/master/dataset/KorQuAD_2.1/train 
    with open(data_path) as file:  
        data = json.load(file)

    ### Cleaning html & Extract QA & Create Prompt
    clean_Q = []
    clean_A = []
    prompt_text = []
    for i in range(len(data)):
        Q = cleanhtml(data[i]['question'])
        A = cleanhtml(data[i]['answer']['text'])
        temp = "## 질문: " + Q + "\n## 답변: " + A + "\n\n"
        clean_Q.append(Q)
        clean_A.append(A)
        prompt_text.append(temp)

    ### Create data dictionary
    data_dict = dict()
    data_dict['question'] = clean_Q
    data_dict['answer'] = clean_A
    data_dict['text'] = prompt_text
    
    ### Transforming Huggingface Format
    dataset = ds.Dataset.from_dict(data_dict)
    print(dataset)

    ### Upload Huggingface
    dataset.push_to_hub("leeseeun/KorQuAD_2.0")

    