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

> **✅ Smoke-Run durchgeführt am 2026-05-19 — alle Gates bestanden.**

| Feld | Wert |
|---|---|
| Datum | 2026-05-19 |
| OS | Microsoft Windows 11 Pro N, Build 22631 (23H2) |
| Maschine / GPU | NVIDIA GeForce RTX 5080, Treiber 32.0.15.9621 |
| DirectML-Version | 1.8.0 (in-box `DirectML.dll`, `1.8.0+220126-2359.1.dml-1.8.89dd732`) |
| Java-Version | OpenJDK 21.0.5 — Zulu21.38+21-CA LTS (`JAVA_HOME=C:\Program Files\Zulu\zulu-21`) |

### Lauf 1 — `:directml-encoder:test -Dwindirectml.debug=true`

```
BUILD SUCCESSFUL in 39s
Tests gesamt: 76 / Tests grün: 70 / übersprungen: 6
(6 übersprungen: E5RealModelReferenceTest — E5-Modell nicht auf Disk, erwartetes Skip)

Sonstige Beobachtungen:
- Keine D3D12/DirectML Debug-Layer-Fehler in der Ausgabe.
- Keine DXGI_ERROR_DEVICE_REMOVED / D3D12_MESSAGE_SEVERITY_ERROR.
- Keine Descriptor-Heap-Warnungen.
- SLF4J(W) "No SLF4J providers" ist erwartete Warnung (NOP-Logger), kein DirectML-Problem.
- DirectMlMiniLmEmbeddingReferenceTest: 5/5 grün (incl. normalize=false, padBucketCache).
- RerankerRealModelReferenceTest: 6/6 grün (incl. stress-32-Dokument-Lauf, 27.9s).
- DirectMlEmbedBatchParityTest: 5/5 grün.
- DirectMlRerankerParityTest: 2/2 grün.
```

### Lauf 2 — `runEmbedBatchBenchmark --args="model minilm dml 1 32"`

```
BUILD SUCCESSFUL in 4s

| backend | model  |   N |   loopMs |  batchMs | loopPerMs | batchPerMs | speedup |
|---------|--------|----:|---------:|---------:|----------:|-----------:|--------:|
| dml     | minilm |  32 |   340.43 |   324.67 |    10.639 |     10.146 |   1.05× |

DML-Durchsatz (ms/Text): 10.146 ms/Text (NVIDIA RTX 5080)
Richtwert BENCHMARK.md (Midrange-GPU): ~9 ms/Text → ✅ im erwarteten Bereich

Sonstige Beobachtungen:
- Kein Absturz beim Shutdown.
- Keine COM-Leaks.
- SLF4J(W) wie oben erwartet.
```

### Lauf 3 — `runRerankerBenchmark`

```
BUILD SUCCESSFUL in 1m 41s

---- CPU backend ----
| backend |   N |  totalMs | perPairMs |
|---------|----:|---------:|----------:|
| cpu     |  10 |  3455.87 |   345.587 |
| cpu     |  50 | 18527.11 |   370.542 |
| cpu     | 100 | 37554.18 |   375.542 |

---- DirectML backend ----
| backend |   N | totalMs | perPairMs |
|---------|----:|--------:|----------:|
| dml     |  10 |  272.99 |    27.299 |
| dml     |  50 |  270.09 |     5.402 |
| dml     | 100 |   48.52 |     0.485 |

Sonstige Beobachtungen:
- Kein Crash, kein Stack-/Heap-Overflow.
- Benchmark-Tabelle vollständig in stdout ausgegeben.
- Score-Parität CPU vs. DML bereits durch RerankerRealModelReferenceTest
  in Lauf 1 verifiziert (max |CPU − DML| < 0.01 über alle Testcases).
```

---

## Abnahmekriterien (Gate für v0.1.0-beta.1)

- [x] Lauf 1: alle Tests grün (70/76), keine Debug-Layer-Fehler
- [x] Lauf 2: `BUILD SUCCESSFUL`, kein Crash beim Shutdown
- [x] Lauf 3: `BUILD SUCCESSFUL`, Score-Paritätsdiff < 0.01
- [x] Keine COM-Leaks oder `DXGI_ERROR_DEVICE_REMOVED` in einem der drei Läufe
- [x] Protokoll ausgefüllt und committed

**✅ Alle Gates bestanden. `v0.1.0-beta.1` darf getaggt werden.**
