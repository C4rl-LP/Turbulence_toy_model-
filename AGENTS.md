
# Modelo de vórtices e estocasticidade espontânea

## Objetivo
Investigar numericamente trajetórias, campos de velocidade e
estocasticidade espontânea em um modelo hierárquico de vórtices 2D.

## Antes de modificar código
- Identifique os módulos diretamente envolvidos na tarefa.
- Consulte as interfaces e os testes existentes.
- Apresente a causa provável antes de propor uma refatoração.
- Prefira a menor alteração que resolva o problema.
- Não modifique o modelo físico sem justificar a mudança.

## Física e métodos numéricos
- Preserve equações, convenções, unidades e parâmetros existentes.
- Distinga erros de implementação de hipóteses físicas.
- Ao analisar convergência, explicite os parâmetros mantidos fixos.
- Preserve sementes aleatórias quando necessário para reproduzir testes.
- Não confunda divergência numérica com estocasticidade espontânea.

## Python e desempenho
- Respeite as interfaces públicas dos módulos existentes.
- Evite cópias desnecessárias de arrays e carregamentos de dados grandes.
- Preserve a compatibilidade com NumPy e CuPy quando aplicável.
- Não introduza dependências sem necessidade.
- Não execute simulações longas sem verificar primeiro um caso pequeno.

## Testes e resultados
- Execute primeiro os testes mais específicos e baratos.
- Informe os comandos executados e os resultados observados.
- Não invente resultados de simulações.
- Não sobrescreva resultados experimentais sem autorização.
- Mantenha caches e dados grandes fora do Git quando apropriado.

## Formato das respostas
- Seja conciso.
- Informe arquivos alterados e justificativa.
- Separe fatos observados, hipóteses e conclusões.
- Se faltar informação essencial, pergunte antes de fazer uma mudança arriscada.
