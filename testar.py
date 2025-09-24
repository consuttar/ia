import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Carregue o modelo BASE (Llama 3.1 original) ---
# Ele vai usar a versão que já está no seu cache, não vai baixar de novo.
model_name = "meta-llama/Llama-3.1-8B-Instruct"

print("Carregando o modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- Carregue os "Ajustes" (LoRA) que você treinou ---
# Aponte para a pasta onde seu modelo treinado foi salvo.
adapter_path = "./modelo-orcamentista-final"

print(f"Carregando o adaptador LoRA de '{adapter_path}'...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# --- Agora, vamos testar! ---
print("\n################ MODELO ORÇAMENTISTA PRONTO ################")
print("Faça sua pergunta. Digite 'sair' para terminar.")

while True:
    # Pegue a pergunta do usuário
    prompt = input("Você: ")
    if prompt.lower() == 'sair':
        break

    # Formate a pergunta no padrão que o modelo espera
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Converta o prompt em "tokens" que o modelo entende
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    # Gere a resposta
    print("Orçamentista:", end="", flush=True)
    outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
    
    # Decodifique a resposta e mostre na tela
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrai apenas a parte da resposta, ignorando o prompt
    response_only = response_text.split("### Response:\n")[-1]
    print(response_only)
    print("-" * 20)

