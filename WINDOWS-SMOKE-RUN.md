# Windows-11 / DirectML – lokaler Smoke-Run-Nachweis

Dieses Dokument beschreibt den **manuellen Windows-11-Smoke-Run**, der
vor dem Tag `v0.1.0-beta.1` auf einer echten Windows-11-Maschine mit
DirectML-fähiger GPU ausgeführt und sein Ergebnis hier eingetragen
werden muss.

> **Hintergrund:** Die GitHub-Actions-CI läuft auf Ubuntu. Die drei
> DirectML-Runs können dort konstruktionsbedingt nicht ausgeführt werden
> (`WindowsBindings.isSupported()` → `false`, kein D3D12-Adapter).
> Daher ist dieser manuelle Schritt Teil der Release-Gate-Kriterien
> (Issue #37 Aufgabe 8, Akzeptanzkriterium „Debug-Layer-Smoke-Test läuft
> ohne neue Fehler").

---

## Voraussetzungen

| Anforderung | Details |
|---|---|
| OS | Windows 11 (22H2 oder neuer empfohlen) |
| GPU | DirectX-12-fähige GPU (NVIDIA, AMD, Intel Arc oder integriert) |
| DirectML | in-box `DirectML.dll` (1.8.0 / FL 5.0 reicht; kein Sidecar-DLL nötig) |
| Java | JDK 21 mit `--enable-preview` (FFM API) |
| Gradle | Wrapper im Repo (`./gradlew`) |
| Modell MiniLM | `model/all-MiniLM-L6-v2/` (via `scripts/download-minilm.ps1`) |
| Modell Reranker | `model/cross-encoder-ms-marco-MiniLM-L-6-v2/` (via `scripts/download-reranker.ps1`) |

Modelle laden (PowerShell, einmalig):

```powershell
.\scripts\download-minilm.ps1
.\scripts\download-reranker.ps1
```

---

## Smoke-Run-Kommandos

Alle Kommandos vom **Repo-Root** ausführen.

### 1 — Unit-Tests mit D3D12/DirectML Debug Layer

```powershell
./gradlew :directml-encoder:test -Dwindirectml.debug=true
```

**Was geprüft wird:**

* Alle JUnit-Tests grün (insbesondere
  `DirectMlMiniLmEmbeddingReferenceTest`, `RerankerRealModelReferenceTest`,
  `DirectMlEmbedBatchParityTest`, `DirectMlRerankerParityTest`).
* Keine D3D12/DirectML Debug-Layer-Fehlermeldungen in der Konsole.
* Keine DXGI-Warnungen über Ressourcen-Lifetime oder Descriptor-Heap-Probleme.
* Kein `DXGI_ERROR_DEVICE_REMOVED` / `D3D12_MESSAGE_SEVERITY_ERROR`.

### 2 — EmbedBatch-Benchmark (DML, MiniLM, Batch-32)

```powershell
./gradlew :directml-encoder:runEmbedBatchBenchmark --args="model minilm dml 1 32"
```

**Was geprüft wird:**

* Lauf endet ohne Exception.
* DirectML-Throughput liegt im erwarteten Bereich für die GPU
  (Richtwert aus `directml-encoder/BENCHMARK.md`: ~9 ms/Text auf
  Midrange-GPU bei N=10–100).
* Kein Absturz im `DirectMlGpuBatch`-Fence-Pfad beim Shutdown.
* Keine COM-Leak-Warnungen nach Programmende.

### 3 — Reranker-Benchmark

```powershell
./gradlew :directml-encoder:runRerankerBenchmark
```

**Was geprüft wird:**

* Lauf endet ohne Exception.
* CPU- und DML-Scores stimmen überein (Score-Differenz < 0.01, wie in
  `RerankerRealModelReferenceTest.scoresAgreeBetweenCpuAndDirectMl`
  geprüft).
* Kein Stack-Overflow / Heap-Overflow im 32-Dokument-Stress-Pfad.
* Benchmark-Tabelle erscheint in `stdout`.

---

## Ergebnis-Protokoll

> **Dieser Abschnitt ist auszufüllen, bevor `v0.1.0-beta.1` getaggt wird.**

| Feld | Wert |
|---|---|
| Datum | _(YYYY-MM-DD)_ |
| Maschine / GPU | _(z. B. „Laptop, NVIDIA RTX 4060 Mobile")_ |
| DirectML-Version | _(aus Konsolenausgabe, z. B. „1.8.0 / FL 5.0")_ |
| Java-Version | _(aus `java -version`)_ |

### Lauf 1 — `:directml-encoder:test -Dwindirectml.debug=true`

```
BUILD SUCCESSFUL / FAILED   (zutreffendes ankreuzen)
Laufzeit: _____ s
Tests gesamt: _____ / Tests grün: _____ / übersprungen: _____

Sonstige Beobachtungen:
```

### Lauf 2 — `runEmbedBatchBenchmark --args="model minilm dml 1 32"`

```
BUILD SUCCESSFUL / FAILED
Laufzeit: _____ s
DML-Durchsatz (ms/Text): _____

Sonstige Beobachtungen:
```

### Lauf 3 — `runRerankerBenchmark`

```
BUILD SUCCESSFUL / FAILED
Laufzeit: _____ s
Max |CPU − DML| Score-Differenz: _____

Sonstige Beobachtungen:
```

---

## Abnahmekriterien (Gate für v0.1.0-beta.1)

- [ ] Lauf 1: alle Tests grün, keine Debug-Layer-Fehler
- [ ] Lauf 2: `BUILD SUCCESSFUL`, kein Crash beim Shutdown
- [ ] Lauf 3: `BUILD SUCCESSFUL`, Score-Paritätsdiff < 0.01
- [ ] Keine COM-Leaks oder `DXGI_ERROR_DEVICE_REMOVED` in einem der drei Läufe
- [ ] Protokoll oben ausgefüllt und in `main` committed / an PR #41 angefügt

Erst wenn alle Häkchen gesetzt sind, darf `v0.1.0-beta.1` getaggt werden.
