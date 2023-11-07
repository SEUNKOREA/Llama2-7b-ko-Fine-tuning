import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

### Load fine-tuning model from local
output_merged_dir = "/home/gcp_leeseeun/llama2/results/final_merged_checkpoint"
model = AutoModelForCausalLM.from_pretrained(
    output_merged_dir, 
    device_map="auto", 
    torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(output_merged_dir)

### Load fine-tuning model from huggingface
### 추가예정

# Specify input
eval_data = "이재용이 누구야?"
text = f"## 질문: {eval_data}\n## 답변: "
# Specify device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Tokenize input text
inputs = tokenizer(text, return_tensors="pt").to(device)
# Get answer
# (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
outputs = model.generate(input_ids=inputs["input_ids"].to(device), 
                         attention_mask=inputs["attention_mask"], 
                         max_new_tokens=50, 
                         pad_token_id=tokenizer.eos_token_id)
# Decode output & print it
print(tokenizer.decode(outputs[0], skip_special_tokens=True))