import pandas as pd
import json
import os
import numpy as np
import traceback

# --- CONFIGURAÇÃO ---
PASTA_COM_ORCAMENTOS = "./meus_orcamentos" 
ARQUIVO_SAIDA_JSON = "dados_treinamento_brutos.json"
NOME_DA_ABA = "Analítico Detalhado" 

# --- LÓGICA PRINCIPAL ---
def processar_planilha(caminho_arquivo):
    print(f"\n--- Iniciando processamento de: {caminho_arquivo} ---")
    try:
        # --- Leitura do Contexto ---
        df_contexto = pd.read_excel(caminho_arquivo, header=None, nrows=3, sheet_name=NOME_DA_ABA)
        nome_obra = str(df_contexto.iloc[1, 2]).strip() # Célula C2
        versao_planilha = str(df_contexto.iloc[1, 3]).strip() if df_contexto.shape[1] > 3 else "N/A"
        
        if not nome_obra or pd.isna(nome_obra) or nome_obra == 'nan':
            print("  ERRO FATAL: Nome da obra não encontrado na célula C2. Pulando arquivo.")
            return []
        contexto_obra = f"Orçamento para a obra '{nome_obra}', versão '{versao_planilha}'."
        print(f"  -> Contexto: {contexto_obra}")

        # --- Leitura dos Insumos ---
        df_insumos = pd.read_excel(caminho_arquivo, header=5, sheet_name=NOME_DA_ABA)
        print(f"  -> {len(df_insumos)} linhas lidas inicialmente.")

        if 'Serviço' not in df_insumos.columns:
            print("  ERRO FATAL: Coluna 'Serviço' não encontrada.")
            return []

        # --- REFINAMENTO 1: Renomear e selecionar colunas ---
        df_insumos.rename(columns={
            'Serviço': 'codigo', 'Descrição': 'descricao', 'Unidade': 'unidade',
            'Quantidade': 'quantidade', 'Preço Unit.': 'preco_unitario', 'Preço Total': 'preco_total'
        }, inplace=True)
        
        colunas_relevantes = ['codigo', 'descricao', 'unidade', 'quantidade', 'preco_unitario', 'preco_total']
        df_insumos = df_insumos[colunas_relevantes]

        # --- REFINAMENTO 2: Remover linhas com dados nulos nas colunas principais ---
        df_insumos.dropna(subset=['codigo', 'quantidade', 'preco_unitario', 'preco_total'], inplace=True)
        print(f"  -> {len(df_insumos)} linhas após remover nulos.")

        # --- REFINAMENTO 3: Remover linhas de resumo/título ---
        # Regra: Se a quantidade é 1.0 e o preço unitário é igual ao preço total, é um resumo.
        # A condição `df_insumos['quantidade'] > 1.0001` é uma forma de manter itens com quantidade 1, mas que não são resumos.
        # A condição `df_insumos['preco_unitario'] != df_insumos['preco_total']` também ajuda.
        condicao_resumo = (df_insumos['quantidade'] == 1.0) & (df_insumos['preco_unitario'] == df_insumos['preco_total'])
        df_final = df_insumos[~condicao_resumo].copy() # O '~' inverte a condição, pegando tudo que NÃO é resumo.
        
        print(f"  -> {len(df_final)} linhas após remover linhas de resumo.")

        if df_final.empty:
            print("  AVISO: Nenhuma linha de insumo final válida encontrada.")
            return []

        dados_extraidos = df_final.to_dict(orient='records')

        for insumo in dados_extraidos:
            insumo['nome_obra'] = nome_obra
            insumo['contexto_obra'] = contexto_obra

        print(f"  -> SUCESSO! {len(dados_extraidos)} insumos REFINADOS extraídos.")
        return dados_extraidos
        
    except Exception as e:
        print(f"  ERRO CRÍTICO INESPERADO ao processar {caminho_arquivo}:")
        traceback.print_exc()
        return []

# --- EXECUÇÃO DO WORKFLOW (sem alterações) ---
if __name__ == "__main__":
    # (O código de execução continua o mesmo de antes)
    todos_os_dados = []
    if not os.path.exists(PASTA_COM_ORCAMENTOS):
        os.makedirs(PASTA_COM_ORCAMENTOS)
        print(f"Criei a pasta '{PASTA_COM_ORCAMENTOS}'. Por favor, coloque suas planilhas de orçamento (Excel) dentro dela e rode o script novamente.")
    else:
        arquivos_excel = [f for f in os.listdir(PASTA_COM_ORCAMENTOS) if f.endswith('.xlsx')]
        
        if not arquivos_excel:
            print(f"Nenhum arquivo Excel (.xlsx) encontrado na pasta '{PASTA_COM_ORCAMENTOS}'.")
        else:
            print(f"Encontrados {len(arquivos_excel)} arquivos Excel para processar.")
            for nome_arquivo in arquivos_excel:
                caminho_completo = os.path.join(PASTA_COM_ORCAMENTOS, nome_arquivo)
                dados_da_planilha = processar_planilha(caminho_completo)
                todos_os_dados.extend(dados_da_planilha)

            if todos_os_dados:
                with open(ARQUIVO_SAIDA_JSON, 'w', encoding='utf-8') as f:
                    json.dump(todos_os_dados, f, indent=2, ensure_ascii=False)
                print(f"\nWorkflow concluído! {len(todos_os_dados)} insumos REFINADOS no total foram extraídos e salvos em '{ARQUIVO_SAIDA_JSON}'.")
            else:
                print("\nWorkflow concluído, mas nenhum dado foi extraído. O arquivo JSON de saída está vazio.")
