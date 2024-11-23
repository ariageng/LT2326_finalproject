import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Union
import logging
import json
from data import premade_training_data 

class SwedishDifficultyGenerator:
    def __init__(
        self,
        base_model: str = "AI-Sweden-Models/gpt-sw3-1.3B",  # or /gpt-sw3-356m
        device_id: int = 2,
        max_gpu_memory: str = "10GB"
    ):
        self.logger = self._setup_logger()
        if device_id is None:
            device_id = torch.cuda.current_device()
        self.device = self._setup_device(device_id)
        self.base_model = base_model
        self.max_gpu_memory = max_gpu_memory
        
        self._setup_model_and_tokenizer()
        
    def _setup_logger(self): 
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_device(self, device_id: int) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id {device_id}")
        torch.cuda.set_device(device_id)  # 设置当前CUDA设备
        return torch.device(f"cuda:{device_id}")
    
    def _setup_model_and_tokenizer(self):
        """Initialize model with 4-bit quantization"""
        self.logger.info(f"Loading model: {self.base_model}")
        
        # Quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model,clean_up_tokenization_spaces=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 修改device_map配置
        current_device = torch.cuda.current_device()
        device_map = {'': current_device}

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={self.device.index: self.max_gpu_memory}
        )
        
    def format_prompt(
        self,
        input_text: str,
        input_level: str,
        target_level: str
    ) -> str:
        return f"""För varje exempelmening och dess svårighetsnivå, skapa en ny mening på
        svenska som har en målsvårighetsnivå.
        Använd nedanstående format:
        Exempelmening:{input_text},
        Svårighetsnivå för exempelmening:{input_level},
        Målsvårighetsnivå:{target_level},
        Generera en mening som matchar målsvårighetsnivån:"""
    
    def prepare_training_data(
        self,
        data: List[Dict[str, str]]
    ) -> Dataset:
        """
        Prepare training data from list of dictionaries containing:
        - input_sentence: str
        - input_difficulty: str
        - target_difficulty: str
        - output_sentence: str
        """
        formatted_data = []
        
        for item in data:
            prompt = self.format_prompt(
                item['input_sentence'],
                item['input_difficulty'],
                item['target_difficulty']
            )
            # Format: prompt + expected output
            formatted_text = f"{prompt}\n{item['output_sentence']}"
            
            # Tokenize the text
            tokenized = self.tokenizer(
                formatted_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors=None  # Important: don't convert to tensors yet
            )
            
            # Create labels (shifted input_ids)
            input_ids = tokenized["input_ids"]
            labels = input_ids.copy()  # For causal language modeling, labels are the same as inputs
            
            formatted_data.append({
                "input_ids": input_ids,
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            })
            
        return Dataset.from_dict({
            "input_ids": [d["input_ids"] for d in formatted_data],
            "attention_mask": [d["attention_mask"] for d in formatted_data],
            "labels": [d["labels"] for d in formatted_data]
        })
    
    def prepare_for_fine_tuning(self):
        """Prepare model for LoRA fine-tuning"""
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=8, # LoRa rank
            lora_alpha=16, # 缩放参数
            target_modules=["c_attn", "c_proj"], #目标模块 attention层和projection投影层
            # c_attn 处理注意力权重的计算
            # c_proj 处理注意力的输出转换
            lora_dropout=0.05, 
            bias="none",
            task_type="CAUSAL_LM" #任务类型：因果语言模型
        )
        
        self.model = get_peft_model(self.model, lora_config)

    def fine_tune(
        self,
        training_data: List[Dict[str, str]],
        output_dir: str = "swedish_difficulty_model",
        num_epochs: int = 1,        # 减少轮数
        batch_size: int = 1,        # 简单的batch size
        learning_rate: float = 1e-6,  # 小一点的学习率
        gradient_accumulation_steps: int = 8
    ):
        """Simple fine-tuning setup"""
        from transformers import TrainingArguments, Trainer
        import gc
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        self.prepare_for_fine_tuning()
        dataset = self.prepare_training_data(training_data)
        
        # 最简单的训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            gradient_checkpointing=True,
            fp16=True,
            optim="adamw_torch_fused",
            report_to="none",
            save_strategy="no",
            logging_steps=1,
            max_grad_norm=0.3,  # Add gradient clipping
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            dataloader_pin_memory=False,  # Disable pinned memory
            no_cuda=False,
            use_cpu=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()
        
        # 保存最终模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


    def generate(
        self,
        input_text: str,
        input_level: str,
        target_level: str,
        max_length: int = 500,  
        temperature: float = 0.7,  
    ) -> str:
        prompt = self.format_prompt(input_text, input_level, target_level)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,        # 限制新token数量
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
    
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text.replace(prompt, "").strip()
            
        return result

# Example usage
if __name__ == "__main__":
    # 这下面三行非常关键，没有在所有code之前写这个手动设置并限制住GPU，Trainner会在后面fine tune的时候出现device问题！！！！
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    current_device = torch.cuda.current_device()

    # Initialize generator
    generator = SwedishDifficultyGenerator(device_id=current_device)
    print("model:", AutoModelForCausalLM.from_pretrained(generator.base_model))
    
    # Example: Generate before fine-tuning
    input_text = "Jag gillar att läsa böcker"
    print("\n=== Generation Before Fine-tuning ===")
    result = generator.generate(
        input_text=input_text,
        input_level="1",
        target_level="5",
        temperature=0.8,  
    )
    print(f"Input (Level 1): {input_text}")
    print(f"Generated (Level 5): {result}")
    
    # Import training data
    training_data = premade_training_data
    
    # Optional: Fine-tune and test
    should_fine_tune = input("\nDo you want to fine-tune the model? (yes/no): ").lower()
    
    if should_fine_tune == 'yes':
        # Fine-tune
        generator.fine_tune(training_data)
        
        # Test after fine-tuning
        print("\n=== Generation After Fine-tuning ===")
        result = generator.generate(input_text, input_level="1", target_level="5")
        print(f"Input (Level 1): {input_text}")
        print(f"Generated (Level 5): {result}")