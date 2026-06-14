# Gemma 3 270M-it — Evaluations-Prompts (MODEL-GEMMA-2)

> **Status:** reine Doku/Probe-Vorbereitung. Keine Runtime, keine Kernel, kein Downloader, keine
> Workbench-/Format-Änderung. Dies ist Schritt 1 aus der WAIT-Empfehlung in
> [`model-candidate-gemma3-270m.md`](model-candidate-gemma3-270m.md): eine **billige externe
> Qualitätsprobe** für `google/gemma-3-270m-it` auf unseren engen CPU-Zielaufgaben.

## Zielprofil (nicht „Qwen ersetzen")

`gemma-3-270m-it` wird **nur** als kleiner CPU-Spezialist geprüft für: kurze Code-Erklärung,
Natural/ADABAS-/Legacy-Code erklären, Pseudocode/Java-Skizzen, strukturierte Extraktion (JSON),
kurze technische Zusammenfassung, DE/EN-Umformulierung. **Kein** allgemeiner Coder-Ersatz.

## Datenquellen (vorhanden im Repo — keine externen Domänendaten)

| Quelle | Verwendung |
|--------|-----------|
| `docs/qwen-smoke-test.md` §3.3 | Natural/ADABAS `EMPLOYEES`-Snippet (Code erklären / nach Java skizzieren / JSON-Extraktion) |
| `docs/qwen-smoke-test.md` §3.1/§3.2 | DE/EN-Zusammenfassungstexte (Summary / Umformulierung) |
| `examples/java21-direct-api/.../DirectApiExample.java` | echtes Java-Snippet (erklären / Pseudocode / Seiteneffekte als JSON) |
| `docs/model-candidate-gemma3-270m.md` §5/§6 | technischer Fließtext (technische Zusammenfassung) |

Alle Snippets sind kurz, unkritisch und stammen aus dem Projekt. Keine sensiblen Daten.

## Verwendung

Diese Prompts werden **außerhalb** unserer Runtime gesampelt (HF/`transformers`), siehe
[`model-candidate-gemma3-270m-external-probe.md`](model-candidate-gemma3-270m-external-probe.md) und
`scripts/probe-gemma3-270m-it.py`. Ergebnisse in
[`model-candidate-gemma3-270m-eval-result-template.md`](model-candidate-gemma3-270m-eval-result-template.md)
eintragen. Optionaler Vergleich (falls lokal vorhanden): `SmolLM2-360M-Instruct`,
`Qwen2.5-Coder-0.5B-Instruct`.

Wiederkehrende Snippets:

```natural id=SNIP-NATURAL
DEFINE DATA LOCAL
1 #EMPLOYEES VIEW OF EMPLOYEES
  2 NAME
  2 FIRST-NAME
  2 PERSONNEL-ID
END-DEFINE
READ #EMPLOYEES BY NAME
  DISPLAY NAME FIRST-NAME PERSONNEL-ID
END-READ
END
```

```java id=SNIP-JAVA
try (RerankerModelHandle reranker = runtime.loadReranker(rerankerConfig)) {
    var results = reranker.rerank("What is machine learning?", List.of(
            "Machine learning is a branch of artificial intelligence.",
            "The weather today is sunny.",
            "Deep learning uses neural networks."));
    for (var r : results) {
        System.out.printf("  index=%d, score=%.4f%n", r.originalIndex(), r.score());
    }
}
```

---

## Gruppe 1 — Code erklären

**P1.1** Explain in 2–3 sentences what this Natural/ADABAS program does. Snippet: `SNIP-NATURAL`.

**P1.2** Was bewirkt `READ #EMPLOYEES BY NAME` in `SNIP-NATURAL`? Antworte in einem Satz auf Deutsch.

**P1.3** In `SNIP-NATURAL`: which database view and which three fields are read? List them.

**P1.4** Explain what the Java block `SNIP-JAVA` does, in 2–3 sentences.

**P1.5** In `SNIP-JAVA`, what is the purpose of the try-with-resources around `reranker`? One sentence.

**P1.6** Erkläre kurz (Deutsch, 2 Sätze), was `reranker.rerank(query, docs)` semantisch tut.

**P1.7** Explain this Java line: `float[] vector = embeddings.embed("Hello, world!");`. What is the return type and what does it represent?

## Gruppe 2 — Code in Pseudocode / Java-Skizze übersetzen

**P2.1** Translate the Natural program `SNIP-NATURAL` into ~8 lines of language-agnostic pseudocode.

**P2.2** Sketch a Java method `void listEmployeesByName()` that is equivalent in intent to `SNIP-NATURAL` (just a skeleton, no real DB API).

**P2.3** Convert `SNIP-JAVA` into 5–7 lines of pseudocode (no Java syntax).

**P2.4** Rewrite `SNIP-NATURAL` as a short SQL `SELECT ... ORDER BY` statement that has the same intent.

**P2.5** Skizziere als Pseudocode, wie man `SNIP-JAVA` so erweitert, dass nur Treffer mit `score > 0.5` ausgegeben werden.

## Gruppe 3 — Strukturierte Extraktion als JSON

**P3.1** From `SNIP-NATURAL`, extract a JSON object: `{"view": "...", "accessType": "...", "fields": ["..."]}`. Output JSON only.

**P3.2** From `SNIP-NATURAL`, list any database side effects as JSON: `{"reads": ["..."], "writes": []}`. Output JSON only.

**P3.3** From `SNIP-JAVA`, extract called methods/subprograms as JSON: `{"calls": ["..."]}`. Output JSON only.

**P3.4** From `SNIP-JAVA`, extract side effects as JSON: `{"io": ["..."], "mutations": [], "externalCalls": ["..."]}`. Output JSON only.

**P3.5** Extract from `SNIP-JAVA` the reranker query and the candidate documents as JSON: `{"query": "...", "documents": ["..."]}`. Output JSON only.

**P3.6** From `SNIP-NATURAL`, produce JSON describing the loop: `{"loopKind": "...", "orderedBy": "...", "displays": ["..."]}`. Output JSON only.

## Gruppe 4 — Technische Zusammenfassung

**P4.1** Summarize in one English sentence:
> The Qwen2.5-Coder series is a code-specific large language model built upon the Qwen2.5
> architecture. It has been pretrained on a large corpus of source code and demonstrates strong
> coding ability. The 0.5B variant is the smallest model in the series and is suitable for
> lightweight, low-latency code generation tasks on CPU.

**P4.2** Fasse in einem deutschen Satz zusammen:
> Die Qwen2.5-Coder-Reihe ist eine auf Code spezialisierte Familie großer Sprachmodelle, die auf der
> Qwen2.5-Architektur basiert. Sie wurde auf einem großen Korpus von Quellcode vortrainiert und zeigt
> starke Programmierfähigkeiten. Die 0.5B-Variante ist das kleinste Modell der Serie und eignet sich
> für ressourcenschonende Code-Generierung auf der CPU.

**P4.3** Summarize in 2 bullet points:
> Gemma's 256k LM-head is the dominant per-token cost on FP32-WARP and sits roughly in Qwen-0.5B's
> range, while Gemma's transformer body is much cheaper. A clear speed win over Qwen-0.5B is therefore
> not guaranteed.

**P4.4** In einem Satz (Deutsch): Worin liegt laut dem Text das Hauptrisiko bei `gemma-3-270m-it`?
> `gemma-3-270m-it` ist von Google explizit als Fine-Tune-Basis / task-specific positioniert
> (instruction following, strukturierte Ausgaben, Extraktion), nicht als starker genereller
> Chat-Assistent.

**P4.5** Give a 3-item TL;DR of: "embedding models map text to fixed-length float vectors; rerankers
score query/document pairs; both run on CPU or DirectML in this project."

## Gruppe 5 — DE/EN-Umformulierung

**P5.1** Rephrase in simpler English (1 sentence): "The 0.5B variant is the smallest model in the
series and is suitable for lightweight, low-latency code generation tasks on CPU."

**P5.2** Übersetze ins Deutsche: "A clear speed win over Qwen-0.5B is therefore not guaranteed."

**P5.3** Übersetze ins Englische: "Die 0.5B-Variante eignet sich für ressourcenschonende
Code-Generierung auf der CPU."

**P5.4** Formuliere höflicher/knapper (Deutsch): "Mach das Modell kleiner, sonst ist es zu langsam."

**P5.5** Rewrite for a release note (concise, English): "we made special_tokens_map.json optional so a
404 no longer aborts the Qwen download."

---

**Prompt-Anzahl:** 28 (Gruppe 1: 7, Gruppe 2: 5, Gruppe 3: 6, Gruppe 4: 5, Gruppe 5: 5).
