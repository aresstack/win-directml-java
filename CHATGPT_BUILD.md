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
