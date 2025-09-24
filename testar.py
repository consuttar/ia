import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- ETAPA 1: Carregue o modelo BASE da MESMA FORMA que no treinamento ---
print("Carregando o modelo base com a configuração de treinamento...")

# Use EXATAMENTE a mesma configuração de quantização do script de treino
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
    llm_int8_enable_fp32_cpu_offload=True 
)

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Carrega o modelo base com a quantização. Isso garante que a estrutura seja idêntica.
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- ETAPA 2: Carregue os "Ajustes" (LoRA) que você treinou ---
adapter_path = "./modelo-orcamentista-final"
print(f"Carregando e aplicando o adaptador LoRA de '{adapter_path}'...")

# Aplica o adaptador ao modelo base já quantizado
model = PeftModel.from_pretrained(base_model, adapter_path)

# --- ETAPA 3: Mova o modelo final para a GPU e coloque em modo de avaliação ---
print("Movendo o modelo final para a GPU e preparando para inferência...")
model = model.to("cuda")
model.eval()

# --- ETAPA 4: Agora, vamos testar! ---
print("\n################ MODELO ORÇAMENTISTA PRONTO ################")
print("Faça sua pergunta. Digite 'sair' para terminar.")

while True:
    prompt = input("Você: ")
    if prompt.lower() == 'sair':
        break

    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    print("Orçamentista:", end="", flush=True)
    with torch.no_grad(): # Desativa o cálculo de gradientes para acelerar a inferência
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True, # Torna a resposta um pouco mais criativa
            temperature=0.6,
            top_p=0.9,
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response_text.split("### Response:\n")[-1]
    print(response_only)
    print("-" * 20)
