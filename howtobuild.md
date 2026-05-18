# Build-Anleitung

## Voraussetzungen

- Windows 11
- JDK 21 (z. B. Azul Zulu 21, Eclipse Temurin 21).
  Die Gradle-Toolchain erwartet `JavaLanguageVersion 21`.

## Build

PowerShell:

```powershell
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-21"   # oder anderer JDK-21-Pfad
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
./gradlew.bat clean build
```

Bash / Git-Bash:

```bash
export JAVA_HOME="/c/Program Files/Zulu/zulu-21"
./gradlew clean build
```

Der Build verwendet Java 21 Toolchain, `--enable-preview` (FFM-Vorschau) und
`--enable-native-access=ALL-UNNAMED` für Tests und `JavaExec`.

## Sidecar starten

```powershell
./gradlew.bat :directml-sidecar:run
```

Der Sidecar spricht JSON-RPC 2.0 über stdin/stdout (eine Nachricht pro Zeile).
Logs ausschließlich auf stderr. Details: `directml-sidecar/PROTOCOL.md`.
