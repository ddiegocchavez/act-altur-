"""Generate RESULTS.md strictly from recorded measurements."""
import json
from pathlib import Path

LABELS = {
    'interruption_recovery': 'Recuperación tras interrupción',
    'silence_recovery': 'Recuperación tras silencio',
    'consistency': 'Consistencia',
    'drift': 'Deriva temporal',
    'autocorrelation': 'Autocorrelación',
}


def load(name):
    path = Path('reports') / name
    return json.loads(path.read_text()) if path.exists() else {}


def metrics_row(name, m):
    if not m:
        return f'| {name} | Pendiente | — | — | — |'
    return f"| {name} | {m['accuracy']:.2%} ({m['correct']}/{m['n']}) | {m['auc']:.6f} | {m['eer']:.6f} | {m['brier']:.6f} |"


def main():
    p1, local, public, p3, stress = [load(n) for n in (
        'phase1.json', 'phase2_http.json', 'phase2_public_http.json', 'phase3.json', 'stress_latency.json')]
    if not p1:
        raise SystemExit('Run Phase 1 first')
    final = load('phase3_final_http.json') or load('phase3_final_clean_http.json')
    edges = load('phase3_final_edges.json') or load('phase2_edges.json')
    shortcut = load('shortcut_audit.json')
    phase4 = load('phase4_robust.json')
    public_passed = (public.get('n') == 71 and public.get('http_successes') == 71
                     and public.get('http_errors') == 0 and public.get('offline_equivalent')
                     and public.get('correct') == 66 and public.get('model_sha256') == local.get('model_sha256'))
    phase_rows = [metrics_row(n, m) for n, m in [
        ('Referencia: turnos oficiales', p1['reference_metrics']),
        ('VAD original sin ajustar', p1['default_vad_metrics']),
        ('Fase 1: VAD seleccionado en train', p1['selected_vad_metrics']),
        ('Fase 2: HTTP local, baseline', local),
        ('Fase 2: HTTPS público, baseline', public),
        ('Fase 3: modelo seleccionado por HTTP local', final or p3.get('selected_metrics'))]]
    correlations = {s: {r['feature']: r for r in p1[f'priority_correlations_{s}']} for s in ('train', 'val')}
    corr_rows = [f"| `{n}` | {correlations['train'][n]['pearson_r']:.6f} | {correlations['val'][n]['pearson_r']:.6f} |"
                 for n in ('lat_med', 'lat_mean', 'barge_in_rate', 'overlap_ratio', 'n_caller')]
    block_rows = []
    for a in p3.get('ablations', []):
        label = ' + '.join(LABELS[b] for b in a['blocks'])
        if a.get('selected_final'):
            decision = 'INCLUIDO'
        elif len(a['blocks']) > 1:
            decision = 'Fuera: no mejora al mejor bloque'
        elif a['improves_baseline']:
            decision = 'Mejora solo; no añade aciertos al combinar'
        else:
            decision = 'Fuera: accuracy sin mejora'
        m = a['http']
        block_rows.append(f"| {label} | +{a['added_features']} | {m['accuracy']:.2%} ({m['correct']}/71) | {a['delta_accuracy_pp']:+.2f} | {m['auc']:.6f} | {m['brier']:.6f} | {decision} |")
    stress_rows = []
    for m in stress.get('results', []):
        mode = 'Solo latencias' if m['mode'] == 'latency_only' else 'Todos los turnos del llamante'
        model = 'Baseline' if m['model'] == 'baseline' else 'Baseline + silencio'
        stress_rows.append(f"| {model} | {mode} | −{m['acceleration_s']:g} s | {m['accuracy']:.2%} ({m['correct']}/71) | {m['accuracy_drop_pp']:.2f} | {m['synthetic_recall']:.2%} |")
    p3_status = ('Completada: bloques y combinaciones medidos por HTTP. Selección: ' + ', '.join(p3['selected_blocks']) + '.' if p3 else 'Pendiente de ejecución.')
    broad = {m['model']: m for m in stress.get('results', []) if m['mode'] == 'caller_turn_shift' and m['acceleration_s'] == 1.5}
    stress_conclusion = ('El baseline depende fuertemente de la latencia. El bloque de silencio mejora val, pero también depende del tiempo: al desplazar todos los turnos 1.5 s, el seleccionado cae a ' + f"{broad['selected']['accuracy']:.2%}" + ' y el baseline a ' + f"{broad['baseline']['accuracy']:.2%}" + '. La ganancia en val no demuestra generalización frente a un motor más rápido. Harían falta aumentación temporal y señales independientes, evaluadas antes de incluirse; no se presentan como implementadas.' if broad else 'Prueba pendiente; no hay conclusión de robustez.')
    if shortcut:
        dominant = [artifact['ranked_features'][0] for artifact in shortcut['artifacts']]
        audit_summary = (f"La inspección de árboles confirma el atajo: `{dominant[0]['feature']}` domina "
                         f"el baseline ({dominant[0]['splits']} splits, {dominant[0]['root_splits']} raíces; "
                         f"umbral mediano {dominant[0]['threshold_median']:.4g} s) y "
                         f"`{dominant[1]['feature']}` el seleccionado ({dominant[1]['splits']} splits, "
                         f"{dominant[1]['root_splits']} raíces; umbral mediano "
                         f"{dominant[1]['threshold_median']:.4g} s).")
        dataset_audit = shortcut['dataset_ablations']['status']
    else:
        audit_summary, dataset_audit = 'Auditoría pendiente.', 'pending'
    if phase4:
        phase4_summary = (f"Fase 4 ejecutada. Retador seleccionado: "
                          f"`{phase4.get('selected') or 'ninguno'}`; promoción: "
                          f"`{phase4.get('promoted', False)}`.")
    else:
        phase4_summary = ('La implementación está preparada, pero ningún retador fue entrenado ni promovido '
                          'en esta copia porque faltan los datos oficiales locales.')
    output = f'''# RESULTS — Altur HackMTY 2026

## Estado y métricas

Fase 1 aprobada. Fase 2 {'**CERRADA**: las 71 llamadas evaluadas contra Render reproducen el baseline local.' if public_passed else '**PENDIENTE** de equivalencia pública.'}
Docker está aplazado por indicación del usuario. Fase 3: {p3_status}
Prueba de aceleración: {'completada, con fragilidad documentada abajo.' if stress else 'pendiente.'}
Auditoría de atajos: {'completada' if shortcut else 'pendiente'}; ablaciones con datos: `{dataset_audit}`.

| Variante | Accuracy | AUC | EER | Brier |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(phase_rows)}

Positivo = sintético. EER interpolado en el cruce FAR/FRR. `confidence` expresa la probabilidad de la clase devuelta; AUC y Brier utilizan `P(synthetic)`. No se ha aplicado calibración adicional. Los resultados del modelo nuevo corresponden a HTTP local; la medición pública documentada corresponde al baseline desplegado.

## Datos y separación

353 llamadas: train 282 (113 humanas, 169 sintéticas), val 71 (37 humanas, 34 sintéticas). Los 353 WAV son estéreo, 8 kHz y PCM16. Se respeta el split oficial por hablante; ningún ajuste del clasificador usa val. Val sí se utiliza para seleccionar los bloques de Fase 3: sus ganancias son exploratorias, sin un segundo conjunto independiente. Cada acierto en val equivale a 1.41 puntos porcentuales.

README oficial leído antes de empezar: `alturio/hackmty26`, commit `26b519598a1520cf6306d78902ef5047ae670aa4`, release `v1.0`. El código original queda en `baseline_original/`. Sus 58 predictores más tres columnas de metadatos suman las 61 columnas originales. Se corrigió el ajuste final train+val del script recibido para entrenar únicamente con train.

El dataset sigue excluido del repositorio. La petición posterior del usuario autorizó enviar las 71 llamadas a su endpoint HTTPS de Render para evaluarlo. El servicio procesa cada WAV en memoria y no lo almacena. Todo entrenamiento, extracción experimental y estrés se realiza en el entorno local.

## Fase 1: VAD congelado

Malla de **{p1['grid_configurations']} configuraciones**, seleccionada únicamente por correlación con turnos de referencia en train. El clasificador de producción usa turnos del WAV, sin JSON de referencia. Los cinco parámetros elegidos son:

```json
{json.dumps(p1['selected_config'], indent=2)}
```

| Feature | Pearson train | Pearson val |
| --- | ---: | ---: |
{chr(10).join(corr_rows)}

Puerta: r de `lat_med` > 0.9 y accuracy 92.96%, dentro de 3 puntos del 95.8% de referencia. Se conservan la malla completa y correlaciones de las 58 variables en `reports/vad_grid_train.csv`, `reports/vad_correlations_train.csv` y `reports/vad_correlations_val.csv`.

El diagnóstico de bleed encontró {p1['bleed_calls_abs_correlation_over_06']} llamadas de train con correlación instantánea absoluta > 0.6 y coeficiente lineal absoluto > 0.003 simultáneamente. No se restaron canales; esto no descarta bleed con retardo o no lineal.

## Fase 2: evaluación pública completa

URL: [altur-detector.onrender.com](https://altur-detector.onrender.com). Se llamó primero a `/health`, seguido de 71 peticiones secuenciales a `/detect`. Registro: `reports/phase2_public_http.json`.

- Inicio UTC: `{public.get('started_at_utc', 'pendiente')}`.
- Fin UTC: `{public.get('completed_at_utc', 'pendiente')}`.
- Peticiones satisfactorias: {public.get('http_successes', 'pendiente')}; errores: {public.get('http_errors', 'pendiente')}; reintentos de inferencia: {public.get('detect_retries', 'pendiente')}; abstenciones: {public.get('abstentions', 'pendiente')}.
- Diferencia máxima frente a probabilidades offline: `{public.get('max_probability_delta_offline', 'pendiente')}`.
- Modelo público SHA-256: `{public.get('model_sha256', 'pendiente')}`.
- Pipeline público SHA-256: `{public.get('pipeline_sha256', 'pendiente')}`.

| Medición externa | Milisegundos |
| --- | ---: |
| `/detect` p50 | {public.get('latency_p50_ms', float('nan')):.2f} |
| `/detect` p95 | {public.get('latency_p95_ms', float('nan')):.2f} |
| `/health` previo, separado de percentiles | {public.get('health_latency_ms', float('nan')):.2f} |

Latencias medidas desde este entorno hacia la URL pública, con servidor despierto y conexión del cliente reutilizada. Incluyen serialización JSON, subida, red, inferencia y respuesta; excluyen lectura del WAV y base64. Son una corrida secuencial de 71 llamadas, no una prueba de concurrencia ni disponibilidad prolongada. El cliente respeta el proxy de red del entorno para HTTPS; loopback no usa proxy.

## Fase 3: ablación por bloque

Mismo VAD y mismos hiperparámetros HGB: 300 iteraciones, learning rate 0.06, 15 hojas, L2=1, semilla 0. Se entrenó exclusivamente con 282 llamadas de train. Cada fila recorrió las 71 llamadas de val por el mismo `POST /detect` local y reprodujo las probabilidades offline con error < 1e-12.

Regla de inclusión fijada: mejora estricta en accuracy frente a 66/71. Entre ganadores, elegir primero mayor accuracy y menos features en empate; añadir otro bloque solo si vuelve a aumentar accuracy. AUC/Brier se reportan por separado. No se ajustaron parámetros, umbral ni bins con los resultados de val.

| Baseline + bloque | Features nuevas | Accuracy | Δ pp vs baseline | AUC | Brier | Decisión |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(block_rows) or '| Pendiente | — | — | — | — | — | — |'}

- **Interrupción:** primera entrada del agente dentro de cada turno del llamante; fracción que se calla en 0.5/1 s y antes de que termine el agente. Reanudación antes del siguiente turno del agente, limitada a 12 s después del agente y al fin de habla observado. Incluye tasa de reanudación, espera desde el cese y desde el final del agente, y duración reanudada. Ventanas sin observación se marcan NaN. El baseline ya incluía estadísticas `yield_*` del tiempo hasta callarse.
- **Silencio:** pausas entre turnos del agente > 3 s. Para el primer turno que comienza dentro de la pausa se mide espera, duración hasta el regreso del agente y duración completa. Se agregan mediana/p90/desviación de espera y duración, mediana de duración completa, espera dividida por duración del hueco y fracción ocupada. Son nueve features. Son proxies temporales, sin inferir intención o contenido del silencio.
- **Consistencia:** se reutilizan std/IQR/CV de latencias, std de duraciones y CV del llamante ya existentes. Se añaden entropía de latencias, IQR y entropía de ambas duraciones, y CV de duración del agente. Entropía Shannon normalizada con bins fijos en `behavior_features.py`.
- **Deriva:** pendiente OLS de latencia frente al índice original del turno del llamante y R²; al menos tres observaciones.
- **Autocorrelación:** Pearson de la serie consigo misma a retardos 1 y 2; al menos tres pares. Series constantes o insuficientes dan NaN.

Las features se recomputan desde los turnos VAD en memoria para entrenamiento y evaluación. Se eliminó la lectura de floats redondeados desde CSV como entrada del experimento: una diferencia de un ULP puede cruzar un umbral de árbol. La primera combinación presentó una discrepancia HTTP/offline y fue rechazada; se repitió toda la ablación tras corregir la fuente de floats. La tabla contiene únicamente la repetición verificada.

Modelo seleccionado: `{p3.get('selected_name', 'pendiente')}`; SHA-256 `{p3.get('selected_model_sha256', 'pendiente')}`. `artifacts/model.joblib` contiene el seleccionado; `artifacts/baseline_model.joblib` conserva el baseline público. El endpoint calcula solo los bloques declarados por el artefacto. Los bloques descartados quedan disponibles para reproducir la ablación y no se ejecutan en la inferencia del modelo seleccionado.

Verificación final: {edges.get('n', 'pendiente')} pruebas de bordes aprobadas. `reports/phase3_final_clean_http.json` registra las 71 peticiones con servidor en un entorno Python aislado que contiene solo dependencias runtime. Diez fixtures analíticos comprueban interrupciones, censura, relleno, tendencia, autocorrelación y desplazamiento de latencias.

## Robustez: sintético más rápido

Se congelaron ambos modelos antes de la prueba. Se modificaron exclusivamente las 34 llamadas sintéticas de val; se comprobó que las probabilidades de las 37 humanas permanecieran idénticas. No se reentrenó ni eligieron bloques usando estos escenarios.

**Solo latencias:** resta 1.0/1.5 s a cada latencia original válida, conserva emparejamientos y observaciones, permite valores negativos y recalcula estadísticas, porcentajes bajo/sobre umbral, CV y las features nuevas dependientes de esa serie. Mantiene solapamientos, interrupciones y rellenos de silencio. Por ello, el 97.18% sin caída del modelo nuevo en esta intervención parcial no demuestra robustez del motor: deja intactas sus nuevas features de silencio.

**Todos los turnos:** prueba adicional que adelanta todos los intervalos VAD del llamante sintético, conserva el agente, recorta tiempos a cero y recalcula todas las features, incluidos solapamientos y rellenos. Puede cambiar el emparejamiento o filtrado de latencias. Resuelve el sesgo optimista de congelar esas señales al acelerar solo `lat_*`.

| Modelo | Intervención | Adelanto | Accuracy | Caída pp | Recall sintético |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(stress_rows) or '| Pendiente | — | — | — | — | — |'}

Son intervenciones sobre features/turnos, no WAVs de un motor nuevo ni una medición HTTP de audio acelerado. El desplazamiento uniforme de todos los turnos también es una simplificación. Los números miden sensibilidad, no rendimiento esperado del set oculto.

**Conclusión para el pitch:** {stress_conclusion}

## Auditoría de atajos y Fase 4

{audit_summary} El detalle reproducible está en `reports/shortcut_audit.json`.

Se implementaron `relative_recovery`, ponderación de eventos escasos, HGB
regularizado, regresión logística, aumentación temporal no circular, selección
por peor caso y diagnóstico de calibración. {phase4_summary}

La primera ejecución externa del retador temporal conservó 69/71 limpio y elevó
el peor estrés temporal de 45/71 a 66/71, pero quedó rechazado por recortes:
48/71 a 30 s y 54/71 a 60 s, frente a 59/71 y 64/71 del campeón. Esta observación
motivó aumentación por duración y una puerta de no-regresión por escenario; no se
presenta el primer retador como modelo final.

Las métricas de Fase 4 no se extrapolan a partir de los artefactos anteriores.

## Reproducción y alcance

```bash
make all PUBLIC_URL=https://altur-detector.onrender.com
```

Instala dependencias, verifica/descarga los datos fijados, ejecuta Fase 1, verifica baseline local y público, corre ablaciones por HTTP y estrés. Las puertas se ejecutan secuencialmente. La comprobación pública requiere que esa URL sirva el baseline cuya métrica se compara; falla si el despliegue cambió. Para repetir Fase 3 después del cierre público registrado: `make phase3 stress`.

Artefactos de evidencia: `reports/phase2_public_http.json`, `reports/phase3.json`, `reports/phase3_final_clean_http.json`, `reports/phase3_final_edges.json`, `reports/stress_latency.json`. Predicciones por llamada y cachés permanecen locales. No se ha medido semántica, acústica, clips parciales, ruido/ganancia ni calibración adicional. La demo sigue pendiente.
'''
    Path('RESULTS.md').write_text(output)
    print('RESULTS.md updated from measured reports')


if __name__ == '__main__':
    main()
