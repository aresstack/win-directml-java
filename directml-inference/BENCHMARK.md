# Benchmark-Harness

Reproduzierbare Performance-Messung für Phi-3 (Issue 22).

## Metriken

| Metrik                | Erläuterung                                 |
|-----------------------|---------------------------------------------|
| `modelLoadMs`         | Kalter Modellladevorgang                    |
| `tokensPerSec`        | Aggregat über die Messphase, Warmup separat |
| `avgLatencyMs`        | Durchschnittliche Generierzeit pro Lauf     |
| `hardware`, `jvmArgs` | für Vergleichbarkeit dokumentiert           |

## Ausführung

```powershell
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-21"
./gradlew.bat :directml-inference:jar
java --enable-preview --enable-native-access=ALL-UNNAMED `
     -cp directml-inference/build/libs/*;directml-windows-bindings/build/libs/*;directml-config/build/libs/* `
     com.aresstack.windirectml.inference.bench.Phi3Benchmark `
     model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128 auto 2 5
```

`auto` versucht zuerst DirectML, fällt sonst auf CPU zurück.

## Konvention

- Warmup-Runs werden **immer separat** ausgeführt, nicht in den Aggregaten gezählt.
- Hardware und JVM-Args werden im Report mitgeschrieben.
- Vergleichsläufe sollten denselben Prompt verwenden und `temperature` fix lassen.

## Embedding-Benchmarks

Sobald die Encoder-Runtime live ist:

- `embed/text/sec` (Single)
- `embed/batch[32]/sec`
- `gpu/cpu`-Vergleich
- GPU-Speichernutzung

