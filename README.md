# Altur: detector de comportamiento conversacional

`POST /detect` clasifica al llamante como humano o sintético a partir de un WAV estéreo de 8 kHz: canal 0 llamante, canal 1 agente. Extrae turnos con un VAD congelado y usa HistGradientBoosting, sin GPU ni servicios de inferencia externos.

**Fase 2 cerrada:** las 71 llamadas de val contra [Render](https://altur-detector.onrender.com) dieron **66/71 (92.96%)**, AUC **0.9849**, Brier **0.0453** y latencias externas **p50 586.69 ms / p95 839.89 ms**. Cero errores y probabilidades equivalentes al baseline local.

**Fase 3:** se probaron cinco bloques por separado y una combinación. El artefacto seleccionado añade únicamente respuesta al silencio: **69/71 (97.18%)**, AUC **0.9905**, Brier **0.0306**, verificado por el mismo endpoint en HTTP local. Tiene 67 features: 58 originales y nueve nuevas. `artifacts/baseline_model.joblib` conserva el baseline público; `artifacts/model.joblib` contiene el seleccionado. La evaluación pública registrada corresponde al baseline, y la del seleccionado corresponde a HTTP local.

**Fase 4 promovida:** la aumentación temporal y por duración elevó el resultado
limpio a **70/71 (98.59%)**, AUC **0.9992** y Brier **0.0197**. Al adelantar
1.5 s los turnos sintéticos obtiene **69/71**, frente a 45/71 del campeón
anterior. En audio real mantiene 70/71 con ganancia ±6 dB y remuestreo
8→16→8 kHz; obtiene 61/71 y 65/71 con recortes a 30 y 60 s. Mejoró al campeón
anterior en los seis escenarios y fue promovido solo después de reproducir
70/71 por HTTP sin abstenciones.

Estos resultados siguen siendo exploratorios: las mismas 71 llamadas de val se
utilizaron anteriormente para seleccionar bloques y no sustituyen una evaluación
oculta independiente. Las tablas, hashes y decisiones están en
[RESULTS.md](RESULTS.md); los repositorios de referencia están en
[RESEARCH.md](RESEARCH.md).

## Servir el modelo

Python 3.12, sin descargar el dataset para inferencia:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python serve.py
```

Escucha en `0.0.0.0:8000` y respeta `PORT`. `/health` devuelve los SHA-256 del modelo y del código de inferencia, el número de features y los bloques activos. Para servir otro artefacto propio:

```bash
MODEL_PATH=artifacts/baseline_model.joblib .venv/bin/python serve.py
```

Petición:

```json
{"audio": "<WAV en base64>", "format": "wav"}
```

Respuesta:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`confidence` es la probabilidad de la clase devuelta. AUC/Brier usan `confidence` si el veredicto es sintético y `1-confidence` en caso contrario. Las probabilidades no se redondean; la calibración adicional sigue pendiente.

Mono, audio menor de 3 segundos, silencio y menos de cuatro turnos útiles devuelven `{"is_synthetic": false, "confidence": 0.5}`. A 0.5 hay abstención: el bool es un valor requerido por el contrato. WAV corrupto, frecuencia incompatible y formato distinto de estéreo PCM16 producen 422. Límites: 32 MiB y 900 segundos. El audio se procesa en memoria y no se refleja en errores.

## Reproducir por puertas

```bash
make all PUBLIC_URL=https://altur-detector.onrender.com
```

Instala dependencias, descarga/verifica los datos fijados, ejecuta la malla de Fase 1, verifica baseline local y público, mide los bloques de Fase 3 y ejecuta estrés. Se detiene ante una puerta fallida. Linux/WSL es necesario para la búsqueda con procesos `fork`. Docker queda aplazado por indicación del usuario.

La evaluación pública de Fase 2 compara con el baseline. Si el host ya sirve otro modelo, se detiene al detectar la diferencia; no cambia el despliegue automáticamente. Tras el cierre público ya registrado se pueden repetir los experimentos con:

```bash
make phase3 stress
```

Para evaluar directamente el baseline público:

```bash
.venv/bin/python evaluate_http.py \
  --url https://altur-detector.onrender.com \
  --allow-remote \
  --output reports/phase2_public_http.json
```

`--allow-remote` permite el envío HTTPS solicitado por el usuario a su servicio. Sin esa opción el evaluador acepta únicamente HTTP loopback. No sigue redirecciones. Los audios, predicciones por llamada y cachés siguen fuera de Git.

Para verificar el modelo seleccionado y sus bordes:

```bash
.venv/bin/python verify_selected.py
```

Para auditar atajos y entrenar retadores sin reemplazar al campeón:

```bash
make audit
make robust
```

La promoción es deliberadamente explícita. El entrenamiento incluye recortes
continuos sin mostrar al modelo los puntos exactos de evaluación 30/60 s. Debe
mejorar el peor estrés temporal, conservar al menos 68/71 limpio y no perder más
de una llamada frente al campeón en ningún escenario. También exige los reportes
`audio_robustness_champion.json` y `audio_robustness_candidate.json` con hashes
coincidentes antes de ejecutar:

```bash
.venv/bin/python phase4_robust.py --workers 6 --promote
```

Con el servidor local activo, las tres familias de robustez de audio se ejecutan
con `make audio-robust`. Los WAV transformados permanecen en memoria.

Para evaluarlo por HTTPS después de desplegarlo, usar como comparación offline las predicciones de la variante seleccionada:

```bash
.venv/bin/python evaluate_http.py \
  --url https://altur-detector.onrender.com \
  --allow-remote \
  --offline reports/phase3_silence_recovery_offline.csv \
  --output reports/phase3_public_http.json
```

El CSV se genera localmente al ejecutar `phase3.py`; no se distribuye. La prueba HTTP en entorno limpio se ejecutó con un servidor que tenía únicamente `requirements.lock` instalado. Para repetirla, crear ese entorno y pasarlo mediante `verify_selected.py --python /ruta/al/entorno/bin/python`.

## Qué se midió

- El VAD se seleccionó entre 540 configuraciones usando solo correlaciones de train. `lat_med` obtuvo Pearson **0.9477** en train.
- Clasificador ajustado exclusivamente con 282 llamadas de train. Las 71 de val respetan el split oficial por hablante. No se hizo CV aleatoria sin IDs de hablante.
- Bloques nuevos: recuperación tras interrupción, respuesta al silencio, consistencia, deriva y autocorrelación. Solo se aceptan aumentos estrictos de accuracy; combinar consistencia con silencio no añadió aciertos.
- Misma extracción en memoria al entrenar y servir. Leer floats de un CSV para el experimento provocó una discrepancia en un umbral de árbol; se corrigió y repitió toda la ablación.
- Estrés con modelos congelados: reducción de latencias y desplazamiento completo de turnos, manteniendo intactas las llamadas humanas. Ninguno representa una evaluación de audio real de un motor nuevo.

Val también se reutilizó para seleccionar bloques y retadores: el 98.59% es exploratorio y requiere confirmación independiente. Se midieron ganancia, remuestreo y clips de 30/60 s; todavía no se midieron semántica, acústica neuronal, codecs/eco, calibración independiente ni demo.

## Render y modelo publicado

`render.yaml` configura runtime Python, dependencias fijadas, `python serve.py`, `/health` y el artefacto incluido. No entrena ni descarga datos en Render. Los despliegues automáticos están desactivados en ese Blueprint; aplicar un commit nuevo al servicio requiere desplegarlo y verificar `/health` y las 71 llamadas contra el artefacto correspondiente. La configuración efectiva del servicio creado desde el Dashboard puede diferir del Blueprint.

El plan gratuito puede suspender la instancia: llamar primero a `/health`. Los percentiles registrados excluyen ese chequeo previo. [Documentación de Render](https://render.com/docs/free).

Código: [ddiegocchavez/act-altur-](https://github.com/ddiegocchavez/act-altur-). Datos y condiciones: [repo oficial](https://github.com/alturio/hackmty26), [release v1.0](https://github.com/alturio/hackmty26/releases/tag/v1.0). Uso exclusivo HackMTY 2026; no redistribuir el dataset ni intentar identificar participantes. `baseline_original/` conserva los archivos recibidos; su script de entrenamiento original no debe ejecutarse porque ajustaba también con val.
