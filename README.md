# AGsGPT + LoRA

Capa de red neuronal con adaptadores LoRA sobre AGs-Core v3.6.

- Evidencia: E2 para el código (`agsgpt/model.py`: Transformer de caracteres de ~0,8 M parámetros + `LoRALinear`, con tests). El modelo base no está preentrenado: genera texto sin sentido hasta entrenarlo. No hay adaptadores entrenados.
- Licencia: MIT (código propio, ver LICENSE). Si se usa un modelo base de terceros, su licencia queda PENDIENTE DE VERIFICACIÓN.
- Referencia: Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.

Autor: Lázaro López Paz, El Prat de Llobregat, Barcelona, España, con asistencia de inteligencia artificial. ORCID: 0000-0001-8601-1963.

## Uso
`pip install -r requirements.txt && pytest`. Mismo código que `app/ags_local/model.py` de [ags-core-v3-5](https://github.com/canelonemf-byte/ags-core-v3-5). Atribuciones en [NOTICE](NOTICE); referencias en [REFERENCIAS.md](REFERENCIAS.md).
