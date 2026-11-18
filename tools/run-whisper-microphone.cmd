@echo off
setlocal
set "ROOT_DIR=%~dp0.."
set "ROOT_BASH=%ROOT_DIR:\=/%"
set "JAVA_HOME=C:\Program Files\Zulu\zulu-21"
set "GRADLE_USER_HOME=%ROOT_DIR%\.chatgpt\gradle-home"
set "PATH=%JAVA_HOME%\bin;%PATH%"
if not exist "%JAVA_HOME%\bin\java.exe" (
    echo Java 21 not found: "%JAVA_HOME%"
    pause
    exit /b 1
)
echo Starting Whisper microphone demo, model in "%ROOT_DIR%\model\whisper-tiny"
"C:\Program Files\Git\bin\bash.exe" "%ROOT_BASH%/gradlew" -p "%ROOT_BASH%" :directml-workbench:whisperMicrophoneDemo --no-daemon --max-workers=1 -Dorg.gradle.jvmargs=-Xmx256m
if errorlevel 1 pause
