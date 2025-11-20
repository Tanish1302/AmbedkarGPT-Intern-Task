import json

with open('test_results.json', 'r') as f:
    results = json.load(f)

# Compare strategies
for strategy, data in results['chunking_strategies'].items():
    print(f"\n{strategy}:")
    agg = data['aggregate_metrics']
    print(f"  Hit Rate: {agg['avg_hit_rate']:.3f}")
    print(f"  ROUGE-L: {agg['avg_rouge_l']:.3f}")
    print(f"  Cosine Sim: {agg['avg_cosine_similarity']:.3f}")