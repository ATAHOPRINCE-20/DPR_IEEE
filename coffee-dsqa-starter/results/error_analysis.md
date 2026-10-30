# Qualitative Error Analysis (5 Selected Benchmark Cases)

This section presents five representative queries illustrating the behavioral differences,
strengths, and failure modes between lexical (BM25) and dense semantic retrieval.

### Case 1: Dense Retrieval Advantage (Semantic Paraphrase / Vocabulary Mismatch)
- **Query ID**: `Q010`
- **Topic**: Wind effects
- **Question**: *"Why does the handbook recommend windbreaks in windy coffee-growing areas?"*
- **Expected Gold Passage**: `UCDA_P005` (Robusta Coffee Handbook, 2.4 Rainfall and Humidity; 2.5 Wind Effects)
  > *"Excessive drought can dehydrate coffee trees, cause leaf fall and increase attacks by pests including red spider mite, leaf miner, twig borer and coffee berry borer; severe stress can also lead to wilting or death. Prolonged excessive rainfall may promote vegetative growth while inhibiting or damaging flowering. Strong winds increase water loss and can break trees, increasing irrigation needs. The handbook recommends establishing windbreaks along plantation borders in especially windy locations."*
- **BM25 Retrieval**: Top-1 = `UCDA_P006` (Rank: 2, RR: 0.500, Latency: 0.31 ms)
  - *BM25 Top-1 Passage*: "Intercropping means growing coffee together with one or more other crops in the same field. While coffee is young, the spaces between rows can be used for selected food crops, including bananas, green pepper, cabbages, tomatoes, soya beans, groundnuts and non-climbing Phaseolus beans. These intercrops should be confined to the central two metres of the inter-row, leaving 0.5 metres clear between the intercrop and coffee trees. The handbook warns against continuously growing beans on the same plots, especially in humid areas, because aphid problems may occur."
- **Dense Retrieval**: Top-1 = `UCDA_P005` (Rank: 1, RR: 1.000, Latency: 20.95 ms)
- **Plausible Technical Explanation**:
  Dense retrieval outperformed BM25 due to vocabulary mismatch. The question expressed concepts using paraphrased terms or descriptive syntax that BM25 penalized due to low exact keyword frequency, whereas the dense bi-encoder correctly mapped the semantic intent into proximity with the target passage.

----------------------------------------------------------------------

### Case 2: Dual Concordance (High Lexical Overlap & Clear Semantic Alignment)
- **Query ID**: `Q001`
- **Topic**: Soil requirements
- **Question**: *"What soil characteristics are recommended for productive Robusta coffee?"*
- **Expected Gold Passage**: `UCDA_P001` (Robusta Coffee Handbook, 2.1 Soil and Land Requirements)
  > *"For strong root establishment and high yields, Robusta coffee needs fertile, well-aerated, well-drained, slightly acidic and deep soil containing humus. The handbook describes ideal soils as volcanic red earth or sandy loams with good structure and organic matter. Waterlogged soils and heavy clay or poorly drained soils should be avoided. The recommended soil pH range is 5.5–6.5; soil below pH 5.5 can limit crop performance and may require liming."*
- **BM25 Retrieval**: Top-1 = `UCDA_P001` (Rank: 1, RR: 1.000, Latency: 0.32 ms)
- **Dense Retrieval**: Top-1 = `UCDA_P001` (Rank: 1, RR: 1.000, Latency: 14.71 ms)
- **Plausible Technical Explanation**:
  Both retrievers succeeded because the query contains salient anchor keywords that directly overlap with the source document, while also forming a coherent conceptual question well-aligned with the dense embedding space.

----------------------------------------------------------------------

### Case 3: Agronomic Regional Nuance & Context Discrimination
- **Query ID**: `Q002`
- **Topic**: Soil pH
- **Question**: *"What soil pH range does the UCDA handbook recommend for Robusta coffee?"*
- **Expected Gold Passage**: `UCDA_P001` (Robusta Coffee Handbook, 2.1 Soil and Land Requirements)
  > *"For strong root establishment and high yields, Robusta coffee needs fertile, well-aerated, well-drained, slightly acidic and deep soil containing humus. The handbook describes ideal soils as volcanic red earth or sandy loams with good structure and organic matter. Waterlogged soils and heavy clay or poorly drained soils should be avoided. The recommended soil pH range is 5.5–6.5; soil below pH 5.5 can limit crop performance and may require liming."*
- **BM25 Retrieval**: Top-1 = `UCDA_P001` (Rank: 1, RR: 1.000, Latency: 0.31 ms)
- **Dense Retrieval**: Top-1 = `UCDA_P001` (Rank: 1, RR: 1.000, Latency: 26.27 ms)
- **Plausible Technical Explanation**:
  Both retrievers succeeded because the query contains salient anchor keywords that directly overlap with the source document, while also forming a coherent conceptual question well-aligned with the dense embedding space.

----------------------------------------------------------------------

### Case 4: Agronomic Regional Nuance & Context Discrimination
- **Query ID**: `Q003`
- **Topic**: Altitude
- **Question**: *"At what altitude range can Robusta coffee be grown according to the UCDA handbook?"*
- **Expected Gold Passage**: `UCDA_P002` (Robusta Coffee Handbook, 2.2 Altitude)
  > *"The recommended altitude range for Robusta coffee production is 900–1,500 metres above sea level. Altitude can influence cup profile: coffee grown at higher elevations tends to be more acidic and complex, while coffee grown at lower elevations tends to have a more intense flavour. The handbook describes Uganda as producing relatively high-altitude Robusta at an average of about 1,200 metres above sea level."*
- **BM25 Retrieval**: Top-1 = `UCDA_P002` (Rank: 1, RR: 1.000, Latency: 0.35 ms)
- **Dense Retrieval**: Top-1 = `UCDA_P002` (Rank: 1, RR: 1.000, Latency: 37.23 ms)
- **Plausible Technical Explanation**:
  Both retrievers succeeded because the query contains salient anchor keywords that directly overlap with the source document, while also forming a coherent conceptual question well-aligned with the dense embedding space.

----------------------------------------------------------------------

### Case 5: Agronomic Regional Nuance & Context Discrimination
- **Query ID**: `Q004`
- **Topic**: Altitude and cup profile
- **Question**: *"How does altitude affect the cup profile of Robusta coffee?"*
- **Expected Gold Passage**: `UCDA_P002` (Robusta Coffee Handbook, 2.2 Altitude)
  > *"The recommended altitude range for Robusta coffee production is 900–1,500 metres above sea level. Altitude can influence cup profile: coffee grown at higher elevations tends to be more acidic and complex, while coffee grown at lower elevations tends to have a more intense flavour. The handbook describes Uganda as producing relatively high-altitude Robusta at an average of about 1,200 metres above sea level."*
- **BM25 Retrieval**: Top-1 = `UCDA_P002` (Rank: 1, RR: 1.000, Latency: 0.26 ms)
- **Dense Retrieval**: Top-1 = `UCDA_P002` (Rank: 1, RR: 1.000, Latency: 32.21 ms)
- **Plausible Technical Explanation**:
  Both retrievers succeeded because the query contains salient anchor keywords that directly overlap with the source document, while also forming a coherent conceptual question well-aligned with the dense embedding space.

----------------------------------------------------------------------
