@echo off
set "JAVA_HOME=C:\Program Files\Zulu\zulu-21"
echo Using JAVA_HOME=%JAVA_HOME%
call gradlew.bat :directml-windows-bindings:test --tests "com.aresstack.windirectml.windows.QwenAttentionShadersGpuTest" -Dqwen.gpu.test=true -i
echo EXIT=%ERRORLEVEL%
