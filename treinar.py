import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# 1. Carregar o Dataset e formatá-lo imediatamente
print("Carregando e formatando o dataset...")

def format_instruction(example):
    # Esta função cria o prompt no formato que o modelo espera
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

dataset = load_dataset("json", data_files="dados_treinamento.json", split="train")
# Aplica a formatação e cria a coluna 'text' que o SFTTrainer usará
dataset = dataset.map(lambda example: {"text": format_instruction(example)})


# 2. Configurar o Modelo com Quantização (para caber na sua GPU)
print("Configurando o modelo...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # Deixa o PyTorch decidir como usar a GPU
    trust_remote_code=True
)
model.config.use_cache = False
# Adiciona otimização para resolver problema de memória, se necessário
model.config.pretraining_tp = 1


# 3. Configurar o Tokenizer
print("Configurando o tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# 4. Configurar o LoRA (os "ajustes" que vamos treinar)
print("Configurando LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# 5. Configurar os Argumentos de Treinamento
print("Configurando os argumentos de treinamento...")
training_arguments = TrainingArguments(
    output_dir="./modelo-orcamentista-final",
    num_train_epochs=10, # Vamos ler o "livro" 10 vezes
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10, # Log a cada 10 passos para não poluir o terminal
    fp16=True, # Usar precisão de 16 bits para acelerar
    push_to_hub=False,
)


# 6. Criar o Trainer (versão corrigida)
print("Inicializando o Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", # Agora esta linha está correta, pois criamos a coluna "text" manualmente
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
)


# 7. Iniciar o Treinamento!
print("################ INICIANDO O TREINAMENTO ################")
trainer.train()


# 8. Salvar o modelo final
print("Salvando o modelo treinado...")
trainer.save_model("./modelo-orcamentista-final")

print("################ TREINAMENTO CONCLUÍDO ################")

