# GPT-compatible build

This repository can publish a rolling GPT-compatible ZIP that contains the source tree and a prepared Gradle cache.

## Local entry point

Run from the repository root:

```bash
bash chatgpt-build.sh
```

The script sets `GRADLE_USER_HOME` to `.chatgpt/gradle-home` and runs Gradle in offline mode. It calls `gradlew` through
`bash`, so the build still works after ZIP extraction even when the executable bit was lost.

If `xvfb-run` is available, the script wraps Gradle with a virtual display. This keeps UI-oriented tests stable on
headless Linux runners.

## Rolling GitHub release

The workflow `.github/workflows/gpt-compatible-release.yml` runs on every push to `main` and can also be started
manually.

It performs these steps:

1. Set up JDK 21.
2. Install `xvfb`, `zip` and `rsync`.
3. Fill `.chatgpt/gradle-home` with the Gradle distribution and Maven dependencies.
4. Package the repository and prepared cache into `win-directml-java-gpt-compatible.zip`.
5. Extract that ZIP into a fresh temporary directory.
6. Remove the executable bit from `gradlew` to simulate lossy ZIP extraction.
7. Verify `bash chatgpt-build.sh` from the extracted ZIP.
8. Replace the old release asset on the fixed `gpt-compatible` release.

The release asset is intentionally overwritten so external GPT sessions can always use the same download target.

## Offline tests (JUnit launcher)

All modules run their tests offline against the prepared `.chatgpt/gradle-home` cache, including
`:directml-config`. The Java-21 modules inherit JUnit from the root `subprojects` convention (a
`org.junit:junit-bom` platform that pins `junit-platform-launcher` to a cached, concrete version). The root
convention skips `java8Module` projects, so any such module (e.g. `directml-config`) must declare the JUnit
deps itself **via the same BOM** — never an unversioned `org.junit.platform:junit-platform-launcher`, which
cannot resolve offline. Keep the BOM version aligned with the root (currently `5.10.2` → launcher `1.10.2`).
