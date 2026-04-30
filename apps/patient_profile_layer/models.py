from __future__ import annotations

# Intencionalmente sem modelos persistentes.
# Esta camada é snapshot-first e escreve artefactos JSONL/Parquet.
# Se no futuro quiseres auditar execuções na base de dados,
# adiciona aqui um model ProfileBuildRun e as respetivas migrações.
