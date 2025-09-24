import json
import random

ARQUIVO_DADOS_BRUTOS = "dados_treinamento_brutos.json"
ARQUIVO_TREINAMENTO_FINAL = "dados_treinamento_final.json"

def formatar_valor(valor):
    """Formata um número como moeda brasileira."""
    try:
        return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "N/A"

def gerar_instrucoes_para_insumo(insumo):
    """
    Recebe um dicionário de insumo e gera uma lista de pares de instrução/resposta.
    Esta função é o "cérebro" da geração de perguntas.
    """
    
    # Extrai e formata os dados do insumo para facilitar o uso
    desc = insumo.get('descricao', 'N/A').strip()
    codigo = insumo.get('codigo', 'N/A')
    obra = insumo.get('nome_obra', 'N/A')
    
    qtd = insumo.get('quantidade', 0)
    un = insumo.get('unidade', 'N/A')
    
    preco_unit = formatar_valor(insumo.get('preco_unitario'))
    preco_total = formatar_valor(insumo.get('preco_total'))
    
    instrucoes = []

    # Template 1: Pergunta direta sobre o custo total
    instrucao1 = f"Qual o preço total para o item '{desc}' (código {codigo})?"
    output1 = f"O preço total orçado para '{desc}' é de {preco_total}."
    instrucoes.append({"instruction": instrucao1, "output": output1})

    # Template 2: Pergunta com contexto da obra
    instrucao2 = f"No orçamento da obra '{obra}', qual foi o valor total alocado para '{desc}'?"
    output2 = f"Para a obra '{obra}', o valor total alocado para o insumo '{desc}' foi de {preco_total}."
    instrucoes.append({"instruction": instrucao2, "output": output2})

    # Template 3: Pergunta sobre detalhes (quantidade e preço unitário)
    instrucao3 = f"Poderia me dar os detalhes de quantidade e preço unitário para '{desc}' na obra '{obra}'?"
    output3 = f"Claro. Para a obra '{obra}', o item '{desc}' teve uma quantidade orçada de {qtd} {un}, com um preço unitário de {preco_unit}."
    instrucoes.append({"instruction": instrucao3, "output": output3})
    
    # Template 4: Pergunta mais aberta, simulando um gerente
    instrucao4 = f"Me dê um resumo do insumo com código {codigo}."
    output4 = f"O insumo de código {codigo} é '{desc}'. No orçamento da obra '{obra}', foram previstas {qtd} {un} a um custo unitário de {preco_unit}, totalizando {preco_total}."
    instrucoes.append({"instruction": instrucao4, "output": output4})

    return instrucoes

# --- EXECUÇÃO DO WORKFLOW ---
if __name__ == "__main__":
    try:
        with open(ARQUIVO_DADOS_BRUTOS, 'r', encoding='utf-8') as f:
            dados_brutos = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{ARQUIVO_DADOS_BRUTOS}' não foi encontrado. Rode o 'gerador_dataset.py' primeiro.")
        exit()
        
    dataset_final = []
    print(f"Iniciando geração de exemplos a partir de {len(dados_brutos)} insumos...")
    
    for insumo in dados_brutos:
        # Para cada insumo, gera um conjunto de perguntas e respostas
        novas_instrucoes = gerar_instrucoes_para_insumo(insumo)
        dataset_final.extend(novas_instrucoes)
        
    # Embaralha o dataset para que o treinamento seja mais eficaz
    random.shuffle(dataset_final)
    
    with open(ARQUIVO_TREINAMENTO_FINAL, 'w', encoding='utf-8') as f:
        json.dump(dataset_final, f, indent=2, ensure_ascii=False)
        
    print(f"\nWorkflow concluído! {len(dataset_final)} exemplos de treinamento gerados e salvos em '{ARQUIVO_TREINAMENTO_FINAL}'.")
    print("Próximo passo: Use este arquivo no script 'treinar.py' para treinar o modelo com os novos dados.")

