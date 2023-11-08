from model_utils import create_bnb_config, load_model, get_max_length, find_all_linear_names, create_peft_config, print_trainable_parameters
from data_utils import preprocess_dataset
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
import os
import torch

def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=0,
            weight_decay=0,
            max_steps=1501,
            # num_train_epochs=15,
            learning_rate=0.0001,
            fp16=True,
            logging_steps=100,
            output_dir=output_dir,
            optim= "adamw_torch_fused", # "paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch training
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    ### Load model from HF and with bitsandbytes config
    model_name = "beomi/llama-2-ko-7b"
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)


    ### Load dataset from HF
    dataset_name = "leeseeun/KorQuAD_2.0"
    dataset = load_dataset(dataset_name, split="train[:2000]")
    print(dataset)


    ### Preprocess dataset
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, dataset)
    print("After Preprocessing")
    print(dataset)
    

    ### Train
    output_dir = "/home/gcp_leeseeun/llama2/results/final_checkpoint"
    train(model, tokenizer, dataset, output_dir)


    ### Merge weights
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = "/home/gcp_leeseeun/llama2/results/final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_merged_dir)


    ### Upload Huggingface
    huggingface_repo = "leeseeun/llama2-7b-ko-finetuning"
    model.push_to_hub(huggingface_repo)
    tokenizer.push_to_hub(huggingface_repo)
