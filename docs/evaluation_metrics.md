# Evaluation Metrics

## BLEC

$$BLEC = 1 - \frac{D_{seq}}{\max(L_{ref}, L_{hyp})}$$

## Swing

$$Swing = \frac{\Delta t_{off}}{T}$$

## Latency

```bash
python -m modcompose.evaluate --latency output.mid
```
