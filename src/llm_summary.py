import os
import logging
import openai
from openai import OpenAI
from modelConfig import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def gerarResumoProntuario(transcricao):
    """
    Recebe a transcrição completa e gera um resumo do prontuário com os seguintes tópicos:

    **Dados do Paciente**
    - Nome do paciente, data da consulta (formato DD/MM/AAAA) e horário (formato HH:MM).
    - Nome do acompanhante (se identificado).

    **Medicação em Uso**
    - Liste os nomes e as doses dos medicamentos que o paciente está utilizando.

    **Exames Recentes**
    - Liste os exames realizados e apresentados durante a consulta.

    **Exames a Marcar**
    - Liste os exames pedidos e, se informado, a data da próxima consulta sugerida.

    **Resumo da Consulta**
    - Resuma os principais problemas relatados, possíveis diagnósticos e pontos de atenção.

    **Sugestão de Diagnóstico**
    - Sugira um diagnóstico para apoio ao médico, indicando exames, procedimentos e pontos de atenção.

    Esta função utiliza a API de ChatCompletion da OpenAI com o modelo definido em model_openai para gerar o resumo.
    """
    prompt = f"""Você é um especialista em medicina e análise de dados clínicos. Abaixo está a transcrição completa de uma consulta médica.
Por favor, gere um resumo do prontuário seguindo estes tópicos:

### Dados do Paciente
- Nome do paciente, data da consulta (formato DD/MM/AAAA) e horário (formato HH:MM).
- Nome do acompanhante (se identificado).

### Medicação em Uso
- Liste os nomes e as doses dos medicamentos que o paciente está utilizando.

### Exames Recentes
- Liste os exames realizados e apresentados durante a consulta.

### Exames a Marcar
- Liste os exames pedidos e, se informado, a data da próxima consulta sugerida.

### Resumo da Consulta
- Resuma os principais problemas relatados, possíveis diagnósticos e pontos de atenção.

### Sugestão de Diagnóstico
- Sugira um diagnóstico para apoio ao médico, indicando exames, procedimentos e pontos de atenção.

Transcrição completa:
{transcricao}
"""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_openai,
            messages=[
                {"role": "system", "content": "Você é um especialista em medicina e análise de dados clínicos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        logging.exception("Erro ao gerar resumo do prontuário:")
        return f"Erro ao gerar resumo: {e}"



