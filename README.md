# TC3---MLET---MODEL-LC

1) Qual é o seu problema?
    - Classifição de empresas listadas em bolsa com base nos valores atrelados ao equity da empresa.
    - No caso os valores selecionados são o ROE, que indica o retorno em relação ao Equity. E a dívida sobre o Equity, que indica o quanto que uma empresa está endividada em relação ao seu Equity
    - Vamos validar a relacão dívida e retorno tendo como base o Equity das empresas
2) Coleta de dados
    - Yahoo finance
    - Os dados são limitados devido a necessidade de buscas um ativo por vez, por isso tivemos que limitar os dados que teremos acesso
3) Armazenamento
    - Não estruturado
        - CSV no S3
4) Analisar
    - Qual é o comportamento do seu dado?
	    - Os dados são atrelados a empresas em relação ao retorno e dívida em relação ao Equity
    - Quais são as particularidades?
	    - Temos dados limitados
	    - Estamos utlizando aprendizado não supervisionado
	    - O plot de dados é em tempo real
	    - Os dados podem ser utilizados para direcionar entendimentos sobre empresas da bolsa
    - Como são as distribuições?
		- Os clusters agrupam empresas com perfis financeiros semelhantes em termos de rentabilidade (ROE) e alavancagem (D/E).
		- Distribuição visual: No gráfico, vemos grupos bem separados, indicando que há empresas com ROE alto e D/E baixo (mais rentáveis e menos endividadas), e outras com ROE baixo e D/E alto (menos rentáveis e mais alavancadas).
		- Tamanho dos clusters: Alguns clusters podem ter poucos membros (outliers), enquanto outros concentram a maioria das empresas.
    - Como são as correlações entre as variáveis?  
		- Correlação esperada: Em geral, empresas com ROE alto tendem a ter D/E mais baixo, pois conseguem gerar lucro sem depender tanto de dívida.
		- No contexto brasileiro: Algumas empresas podem apresentar ROE alto mesmo com D/E elevado, indicando uso eficiente de alavancagem.
		- Nos clusters: Se o gráfico mostra pontos dispersos, a correlação pode ser fraca; se há uma tendência (linha), a correlação é mais forte.
    - Existem sazonalidades?
		- Sazonalidade não é diretamente observável apenas com ROE e D/E anuais, pois são métricas de resultado e estrutura de capital.
		- Para detectar sazonalidade: Seria necessário analisar dados trimestrais ou mensais, ou incluir variáveis como receita, lucro operacional ao longo do tempo.
		- No contexto atual: Não é possível afirmar sazonalidade apenas com os dados apresentados.
    - Análises Estatísticas
		- **Médias e medianas:** Cada cluster tem sua média e mediana de ROE e D/E, que podem ser usadas para caracterizar o perfil do grupo.
		    - Exemplo: Cluster 0 pode ter ROE médio de 0.15 e D/E médio de 0.3.
		- **Desvios padrão:** Indicam a dispersão dos valores dentro de cada cluster. Clusters com desvio baixo são mais homogêneos.
		- **Outliers:** Empresas muito distantes dos centros dos clusters (ex: ROE muito alto ou D/E muito alto) podem ser outliers, e aparecem isoladas no gráfico.
		- **Boxplot:** Pode ser usado para visualizar a distribuição de ROE e D/E em cada cluster e identificar outliers.
	- Resumo Contextual
		- **Empresas com ROE alto e D/E baixo** são geralmente mais saudáveis financeiramente.
		- **Empresas com D/E alto** podem estar mais alavancadas, correndo mais risco, mas podem ter ROE elevado se usam bem a dívida.
		- **Clusters ajudam a identificar grupos de empresas com estratégias financeiras semelhantes**, facilitando comparações e decisões de investimento.
5) Processamento dos dados (se necessário)
	- **Enriquecimento dos dados** - Com os dados básicos dos tickers, você buscamos as métricas financeiras como ROE, Debt-to-Equity, lucro, patrimônio. Com esses dados realizamos a normalização e tratamento de dados
	 - **Enriquecimento de chave x valor** - Isso é feito ao montar o DataFrame com as informações coletadas para cada empresa.
	 - **Cálculo** - No seu caso, já utiliza cálculos como ROE e Debt-to-Equity, que são indicadores compostos.

 ```python
 # 1. Coleta dos dados financeiros
data = fetch_financial_metrics(tickers)

# 2. Conversão para numérico e tratamento de nulos
num_cols = ['roe', 'debt_to_equity']
for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col].fillna(data[col].median(), inplace=True)

# 3. Escalonamento dos dados (normalização entre 0 e 1)
preprocessor = create_preprocessor(num_cols=num_cols)
scaled_features = preview_transformation(preprocessor, data[num_cols])

# 4. Montagem do DataFrame processado
df_processed = pd.DataFrame(scaled_features, columns=num_cols)
df_processed['ticker'] = data['ticker'].values
 ```


- **Coleta:** Busca dos dados financeiros dos tickers selecionados.
- **Tratamento:** Conversão dos dados para formato numérico e preenchimento dos valores nulos com a mediana.
- **Escalonamento:** Normalização dos dados para garantir que todas as variáveis estejam na mesma escala.
- **Preparação:** Criação do DataFrame final para ser usado na modelagem.

6) Modelagem

```python
# 5. Escolha e aplicação do modelo de agrupamento (KMeans)
results = apply_kmeans_and_visualize(df_processed, data, optimal_k=4)
```
- **Escolha do modelo:** Utilização do algoritmo KMeans para agrupar empresas com perfis financeiros semelhantes.

- Escolher o modelo
Você escolheu o KMeans para agrupar empresas com perfis financeiros semelhantes. Outros modelos poderiam ser testados, como DBSCAN ou Gaussian Mixture, dependendo do objetivo.

- Testar os modelos (métricas de avaliação)
Para clustering, uma métrica comum é a inércia (usada no método do cotovelo) e o silhouette score. Você pode comparar diferentes valores de K e diferentes algoritmos para ver qual separa melhor os grupos.

- Comparar as versões do modelo
Teste diferentes configurações (número de clusters, features usadas, algoritmos) e compare as métricas de avaliação. Escolha a versão que melhor representa os grupos de interesse.

- Interpretar os resultados
Analise os clusters formados: quais empresas estão juntas? Quais métricas são mais relevantes para a separação? Use estatísticas descritivas e visualizações para entender o perfil de cada grupo.

- Escolha do modelo "campeão"
O modelo campeão é aquele que melhor separa os grupos de acordo com o contexto do negócio e as métricas de avaliação. No seu caso, pode ser o KMeans com determinado valor de K que apresenta clusters bem definidos e interpretáveis.


7) Deploy
    - Está sendo utilizado pelo Superset para validação e tomada de decisão com base nos dados coletados e categorizados


## **Conclusões dos Dados**

### **1. Distribuição dos Clusters**

- Os gráficos mostram que as empresas brasileiras foram agrupadas em diferentes clusters de acordo com seus indicadores financeiros (ROE e Debt-to-Equity).
- Cada cor representa um grupo de empresas com perfil financeiro semelhante.
- A separação entre os grupos indica que o modelo conseguiu identificar padrões distintos de rentabilidade e alavancagem.

### **2. Perfis dos Grupos**

- **Clusters com ROE alto e D/E baixo:** Empresas mais rentáveis e menos endividadas, geralmente consideradas mais saudáveis financeiramente.
- **Clusters com D/E alto:** Empresas mais alavancadas, que podem apresentar maior risco, mas também potencial de retorno se utilizam bem a dívida.
- **Clusters intermediários:** Empresas com indicadores médios, formando grupos de perfil misto.

### **3. Estatísticas dos Clusters**

- As análises estatísticas (média, mediana, desvio padrão) mostram que cada grupo tem características próprias.
- Os boxplots evidenciam a distribuição dos indicadores dentro de cada cluster e ajudam a identificar possíveis outliers.
- A contagem de outliers por cluster indica que alguns grupos possuem empresas com valores extremos, que podem merecer atenção especial.

### **4. Interpretação**

- O modelo KMeans foi eficiente para separar empresas em grupos com perfis financeiros distintos.
- Essa segmentação pode ser usada para comparar empresas, identificar oportunidades de investimento ou monitorar riscos.
- O tratamento dos dados (normalização, preenchimento de nulos) foi fundamental para garantir resultados confiáveis.

---

**Resumo para apresentação:**

- O agrupamento permitiu identificar grupos de empresas com características financeiras semelhantes.
- As estatísticas e visualizações facilitam a análise dos perfis e dos riscos de cada grupo.
- O processo pode ser replicado para outros mercados ou indicadores, auxiliando decisões estratégicas e análises financeiras.