# Investigación técnica: voz sintética en llamadas bancarias

Revisión al 12 de septiembre de 2026. La conclusión práctica es conservar el
comportamiento conversacional como señal principal. La literatura reciente
confirma que la generalización entre motores, codecs y ataques sigue siendo el
problema difícil de los detectores acústicos; no justifica sustituir una señal
barata y alineada con el reto por un modelo grande entrenado con 282 llamadas.

## Fuentes y repositorios que sí vale la pena conocer

| Recurso | Qué aporta | Decisión para Altur |
| --- | --- | --- |
| [ASVspoof 5](https://github.com/asvspoof-challenge/asvspoof5) | Baselines oficiales RawNet2/AASIST, métricas de coste y fusión. La evaluación incluye ataques y codecs no vistos. | Referencia de metodología. No entra al camino crítico. |
| [Resultados ASVspoof 5](https://arxiv.org/abs/2601.03944) | Evidencia reciente de degradación con ataques adversarios, compresión neural y evaluación cruzada; también separa discriminación de calibración. | Refuerza peor-caso, pruebas de canal y calibración explícita. |
| [AASIST](https://github.com/clovaai/aasist) | Baseline acústico end-to-end con atención espectro-temporal. | Descartado: GPU, dominio distinto y el experimento local previo dio ~49 %. |
| [AI-Synthesized-Voice-Generalization](https://github.com/Purdue-M2/AI-Synthesized-Voice-Generalization) | Implementación oficial AAAI 2025 orientada a vocoders no vistos y a reducir atajos de contenido/dominio. | Buena referencia futura; demasiado grande para el hackathon y no resuelve por sí sola el diálogo. |
| [AttM Interspeech 2024](https://github.com/pandarialTJU/AttM_INTERSPEECH24) | Combina capas ocultas de WavLM para anti-spoofing. | Solo experimento posterior y aislado; no añadir antes de cerrar robustez conductual. |
| [RawNetLite](https://github.com/adipiz99/rawnetlite) | Detector acústico compacto. | Posible desempate futuro si se valida fuera de dominio; no es el núcleo. |
| [Silero VAD](https://github.com/snakers4/silero-vad) | VAD CPU/ONNX con soporte de 8 y 16 kHz. | Único reemplazo razonable si reduce el error absoluto de fronteras en train y mueve robustez. |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | VAD, solapamiento y diarización completos. | Excesivo para dos canales ya separados y para el presupuesto de latencia. |

## Qué se adopta ahora

1. Señales relativas dentro de cada llamada: dispersión, diferencias entre
   respuestas y recuperación comparada con la mediana propia.
2. Medianas de eventos ponderadas por su número de observaciones.
3. Aumentación temporal continua; se excluyen vecindarios de los puntos fijos
   de evaluación para no entrenar sobre el examen.
4. Selección por el peor caso entre limpio y −0.5/−1.0/−1.5 s, con una puerta
   mínima de 68/71 en limpio.
5. Diagnóstico Platt/ECE/log-loss separado de la selección y nunca desplegado
   tras ajustarse y medirse sobre las mismas 71 llamadas.
6. Robustez de audio limitada a ganancia, remuestreo telefónico y recortes.

## Qué queda fuera

AASIST, ECAPA, WavLM, marcas de agua, diarización completa y una interfaz grande.
No son malas tecnologías; son una mala asignación de tiempo y datos para este
brief. La semántica dirigida sigue siendo la única señal complementaria prevista,
y solo se intenta después de cerrar despliegue y robustez.
