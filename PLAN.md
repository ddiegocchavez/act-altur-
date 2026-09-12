> Actualización del usuario: usar `https://github.com/Chaarliee06/altur-detector` y aplazar Docker. La puerta de Fase 2 conserva equivalencia HTTP y verificación del endpoint público con Python; Docker no la bloquea. Las demás puertas y la restricción de procesar el audio localmente permanecen vigentes.

> Plan robusto aprobado: primero publicar y verificar el campeón 69/71; después
> auditar atajos, entrenar `relative_recovery` con perturbaciones continuas,
> seleccionar por peor caso, diagnosticar calibración y ejecutar únicamente las
> pruebas de ganancia, remuestreo y recorte. La semántica permanece al final.

## Tarea

Dada una llamada telefónica entre un llamante y el agente bancario de IA de Altur,
decidir si el llamante es **humano** o **voz sintética**. Entregable:
`POST /detect`, recibe WAV estéreo 8 kHz en base64 (canal 0 = llamante,
canal 1 = agente), devuelve `{"is_synthetic": bool, "confidence": float}`.

## Criterios de calificación (del brief oficial)

1. **Robustez** ante hablantes, motores de voz y condiciones de llamada no vistos.
2. **Originalidad**: ¿usa señales más allá de un clasificador de audio de estantería?
3. **Profundidad técnica**: ¿está bien ejecutado y el equipo entiende *por qué* funciona?
4. **Viabilidad**: ¿podría un banco desplegarlo sobre audio telefónico real?
5. **Latencia**: ¿qué tan rápido decide con confianza razonable?

El brief dice textualmente: *"depth beats breadth. One signal, done well, will
outscore three that half-work."* **Esta es la restricción que manda sobre todo el
plan.** No construyas cuatro familias de señales a medias.

## Estrategia elegida

Señal principal: **comportamiento conversacional**. El brief la describe así: los
humanos se recuperan de interrupciones y silencios de forma instantánea y
desordenada; las máquinas se recuperan de forma consistente, y esa consistencia es
la señal.

Esto responde directamente al criterio de originalidad, corre en CPU en
milisegundos (latencia y viabilidad), y no depende del vocoder concreto del
dataset (robustez).

## Estado actual — ya hecho, no rehacer

Baseline que usa **solo tiempos de turnos**, sin audio. Archivos: `features.py`
(61 features), `train.py`, `vad.py`, `app.py`. Si no están en el repo, correr
`python setup_baseline.py` primero.

Val (hablantes disjuntos, n=71): **accuracy 0.958, AUC 0.981, EER 0.054**.
Calibrado: EER 0.027, Brier 0.032.

Acierto según segundos de llamada disponibles:

| segundosaccuracyAUC |       |       |
| ------------------- | ----- | ----- |
| 20                  | 0.866 | 0.913 |
| 30                  | 0.884 | 0.966 |
| 45                  | 0.915 | 0.950 |
| 60                  | 0.915 | 0.978 |
| 90                  | 0.944 | 0.995 |
| completa            | 0.958 | 0.981 |

Features con mayor separación (Cohen's d en train):

| FeatureHumanoSintéticod                 |        |        |       |
| --------------------------------------- | ------ | ------ | ----- |
| `lat_med` latencia mediana de respuesta | 1.17 s | 2.87 s | -1.76 |
| `lat_under_10` % respuestas < 1 s       | 38%    | 15%    | 1.34  |
| `lat_over_25` % respuestas > 2.5 s      | 21%    | 51%    | -1.25 |
| `barge_in_rate` interrumpe al agente    | 31%    | 15%    | 1.01  |
| `overlap_ratio`                         | 0.154  | 0.059  | 0.97  |

Dato de control: la duración media es casi idéntica entre clases (149.9 s vs
146.5 s), no hay atajo trivial.

---

## FASE 1 — Validar el VAD (bloqueante)

En la evaluación no hay `turns.json`, solo el WAV. Todo depende de reproducir esa
segmentación desde el audio.

1. Corre `vad.py:turns_from_wav()` sobre los 353 WAV.
2. Extrae features con turnos del VAD y con turnos de referencia. Compara.
3. Reporta correlación de Pearson por feature: `lat_med`, `lat_mean`, `barge_in_rate`, `overlap_ratio`, `n_caller`.
4. Búsqueda en malla sobre `thresh_db`, `min_speech`, `min_sil` y el margen sobre el piso de ruido. No lo ajustes a ojo.
5. Si el VAD por energía no alcanza, prueba `webrtcvad` (agresividad 1-3) o Silero. **Ojo con el cruce entre canales**: si el agente se filtra al canal 0, el VAD inventa turnos del llamante y `barge_in_rate` se contamina. Si detectas bleed, resta al canal 0 una versión escalada del canal 1 antes del VAD, estimando la escala por mínimos cuadrados en los tramos donde solo habla el agente.

**Puerta:** `lat_med` del VAD correlaciona r > 0.9 con la referencia, y el modelo
entrenado sobre features-de-VAD queda dentro de 3 puntos del 0.958. Ese número es
tu baseline real.

## FASE 2 — Endpoint desplegado

1. Reentrena con features derivadas del VAD, no de los turnos de referencia.
2. Script de evaluación que corra las 71 llamadas de val **por HTTP** y reporte accuracy, AUC, EER, Brier, y latencia de respuesta p50/p95 del endpoint. Ese script es el medidor de todo lo que sigue.
3. Bordes sin reventar: audio corto, mono, silencio, menos de 4 turnos detectados. En esos casos `confidence: 0.5`, no adivinar.
4. **Despliega a un host estable** (Railway, Render, Fly.io). El jurado corre su benchmark en vivo durante 15 minutos y el endpoint debe estar accesible todo ese rato. No dependas del wifi del evento. Ten ngrok como respaldo y un `/health` para verificar antes de que lleguen.
5. Dockeriza.

**Puerta:** endpoint público respondiendo, reproduciendo la métrica offline.
Desde aquí ya hay entregable. Todo lo demás es mejora.

## FASE 3 — Profundizar en la señal principal

El brief nombra el mecanismo: *"Humans recover from these instantly and messily.
Machines recover consistently, and consistency is a signal."* Las features
actuales apenas rozan eso. Ve más profundo.

1. **Recuperación tras interrupción.** Cuando el agente arranca encima del llamante: ¿cuánto tarda en callarse? ¿reanuda? ¿cuánto tarda en reanudar? ¿repite palabras del tramo interrumpido (requiere transcripción)? Un humano se atropella y repite; una máquina reanuda limpio o descarta el turno entero.
2. **Recuperación tras silencio del agente.** Ya existe `fill_rate`. Profundiza: cuánto espera antes de llenar el silencio, distribución de esas esperas, si el relleno es corto ("¿bueno?") o un turno completo.
3. **Consistencia como señal explícita.** Construye features de *dispersión*, no de tendencia central: desviación, IQR, coeficiente de variación y entropía de las latencias, de las duraciones de turno y de los tiempos de recuperación. La hipótesis del brief es que la máquina es demasiado regular. Mide la regularidad directamente.
4. **Deriva temporal.** ¿Cambia la latencia a lo largo de la llamada? Un humano se relaja o se impacienta; un pipeline mantiene su perfil. Ajusta una regresión de latencia contra índice de turno y usa la pendiente y el residuo.
5. **Autocorrelación** de la serie de latencias. La consistencia deja huella ahí.

Cada bloque pasa por el script de la Fase 2. Si no sube la métrica, fuera.

**Puerta:** documenta qué features nuevas aportaron y cuáles no. Saber por qué
algo *no* funcionó también puntúa en profundidad técnica.

## FASE 4 — Una señal complementaria, bien hecha

Elige **una**, no dos. La semántica es la más alineada con el brief, que dice que
el agente pregunta por cosas que no existen y que una persona responde "no tengo
eso" mientras un modelo de lenguaje tiende a inventar una respuesta.

1. Transcribe canal 0 y canal 1 con `faster-whisper` (`small` o `medium`, español, `word_timestamps=True`, `condition_on_previous_text=False`). Upsample a 16 kHz. Cachea en disco.
2. Detecta los turnos del agente que preguntan por algo inexistente, y clasifica la respuesta del llamante en **niega / pide aclaración / inventa**. Empieza con reglas sobre patrones ("no tengo", "¿cuál?", "no sé de qué", "nunca") y mide.
3. Complementarias baratas: tasa de disfluencias ("este", "mmm", "o sea", falsos inicios, repeticiones) y varianza de palabras por minuto entre turnos.
4. Mide las tres variantes por separado: solo timing, solo semántica, combinado.

**Puerta:** el bloque semántico sube la métrica de forma medible, o se descarta
completo. No lo dejes dentro "porque suena bien": cuesta latencia, y la latencia
se califica.

**La acústica queda fuera por diseño.** Es el eje donde el criterio de
originalidad penaliza, requiere GPU (latencia y viabilidad), y sobreajusta al
vocoder del dataset (robustez). Si sobra tiempo al final, entrénala aparte y
úsala solo como desempate cuando el modelo principal quede cerca del umbral.

## FASE 5 — Robustez, que es el criterio #1

1. **Aumentación**: reentrena añadiendo copias con ruido de canal, cambio de ganancia, y recortes (primeros 30 s, 60 s, ventana aleatoria).
2. **Prueba de estrés del argumento**: simula un motor sintético más rápido restando 1.0 y 1.5 s a las latencias de las llamadas sintéticas de val. Mide la caída. Esto te dice qué tan frágil es depender de la latencia absoluta y justifica las features de consistencia de la Fase 3. **Llévalo al pitch**: demuestra que pensaste en el motor que aún no existe.
3. **Calibración** isotónica o Platt sobre val. El brief dice que `confidence` premia sistemas bien calibrados. Reporta Brier y curva de confiabilidad.
4. **Análisis de errores**: lista los fallos y busca el patrón. Si los falsos negativos son todos sintéticos rápidos, dilo y explica qué lo mitiga.
5. **Curva latencia-acierto** actualizada con el modelo final. Es evidencia directa para el criterio de latencia.

## FASE 6 — Demo de 60 segundos

El jurado tiene 15 minutos y corre su benchmark en paralelo. La demo debe explicar
el *por qué* en un minuto.

- Timeline de los dos canales, uno sobre otro.
- Cada latencia de respuesta anotada y coloreada contra el umbral.
- Interrupciones, solapamientos y recuperaciones marcadas.
- Veredicto con confianza y **las 3 features que más pesaron en esa llamada** (SHAP o contribución del modelo). Que se vea el razonamiento, no solo el número.
- Botón para cargar un ejemplo humano y uno sintético lado a lado. El contraste visual de las latencias es el argumento entero.
- La curva latencia-acierto como gráfica fija.

La demo consume el mismo `POST /detect` que evalúa el jurado. Si la demo corre,
el endpoint corre.

## Restricciones

- Nunca mezcles train y val en validación cruzada aleatoria. Hablantes disjuntos siempre.
- Cada feature nueva pasa por el script de evaluación. Si no sube, fuera.
- `make all` reproduce todo de cero.
- `RESULTS.md` con una tabla por fase y su métrica, para el pitch.
- Nada de subir audio a servicios externos: el dataset es solo para HackMTY 2026, no se redistribuye, y no se intenta identificar a nadie.

## Guion del pitch (redactar en `README.md`)

1. La latencia de respuesta separa las clases con d = -1.76. Aquí está el gráfico.
2. No es un clasificador de audio: es el comportamiento conversacional. Por eso generaliza a motores de voz que no hemos visto.
3. Corre en CPU en milisegundos, sin GPU, sin enrolamiento previo, sin guardar biometría de voz. Un banco lo despliega sobre su audio actual.
4. Decide con 88% de acierto a los 30 segundos de llamada. Sirve en tiempo real, no solo forense.
5. Esto es lo que pasa si el motor sintético se vuelve más rápido, y esto es lo que lo mitiga.


## Instrucción posterior del usuario: evaluación pública y bloques temporales

El servicio público es https://altur-detector.onrender.com. El usuario autorizó expresamente evaluar las 71 llamadas de val contra esa URL HTTPS, despertando primero /health. Esta petición habilita ese envío al servicio del equipo; el entrenamiento y los experimentos permanecen locales. Docker sigue aplazado. La puerta de Fase 2 se cierra al reproducir el 92.96% local por HTTPS, sin exigir la comprobación Docker ni el soak anterior de 900 segundos.

La Fase 3 de este incremento es temporal: recuperación de interrupción, recuperación de silencio, consistencia, deriva y autocorrelación. Evaluar los bloques por separado con el VAD congelado y el mismo endpoint; dejar fuera los que no mejoren accuracy. Después simular latencias sintéticas menores en 1.0 y 1.5 segundos, actualizar RESULTS.md y hacer push. Semántica, acústica y demo no forman parte de este incremento.

## Incremento robusto predeclarado

`audit_shortcuts.py` inspecciona umbrales de los artefactos y, cuando existe la
caché oficial, entrena ablaciones solo-agente, solo-llamante, solo-interacción,
solo-`silence_fill_wait_med` y solo-missingness. `phase4_robust.py` compara el
campeón con HGB regularizado y logística usando perturbaciones reproducibles.

Los desplazamientos de entrenamiento se muestrean continuamente en 0–2 s, pero
se excluyen vecindarios de ±0.1 s alrededor de 0.5, 1.0 y 1.5 s. La promoción
requiere al menos 68/71 limpio y superar estrictamente al campeón en el peor
caso de limpio/−0.5/−1.0/−1.5 s. Sin `--promote`, nunca reemplaza el artefacto.
Platt, ECE y log-loss son un diagnóstico in-sample declarado y no intervienen
en selección ni despliegue. `audio_robustness.py` limita el alcance a las tres
familias aprobadas y procesa todas las transformaciones en memoria.
