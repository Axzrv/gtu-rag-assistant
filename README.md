# PGU-RAG
## Evaluation (Ragas)

Golden set: 10 вопросов по документам ГТУ/ПГУ.

| Метрика | Baseline | + Reranker | Δ |
|---------|----------|-----------|-----|
| Faithfulness | 0.880 | 0.935 | +0.055 |
| Answer Relevancy | 0.628 | 0.785 | +0.157 |
| Context Precision | 0.539 | 0.707 | +0.168 |
| Context Recall | 0.950 | 0.875 | −0.075 |

**Вывод:** reranker `bge-reranker-v2-m3` повысил precision на 0.17 
при небольшом снижении recall — это ожидаемый обмен, 
при котором общее качество ответов выросло.