DirectML Workbench for Windows
==============================

Requirements
------------

- Windows 11
- Java 21 or newer
- The model files downloaded through the Workbench UI

Important
---------

Do not rely on `java` from PATH unless you know it points to Java 21 or newer.
Some Windows machines still have Java 8 first on PATH. In that case this command
will fail even though Java 21 is installed elsewhere:

    java --enable-preview --enable-native-access=ALL-UNNAMED -jar directml-workbench-all.jar

Recommended start
-----------------

Use the included PowerShell launcher:

    powershell -ExecutionPolicy Bypass -File .\directml-workbench.ps1

The launcher checks the selected Java runtime and stops with a clear message if
it is older than Java 21.

Manual start with explicit Java 21
----------------------------------

You can also call the Java 21 executable directly, for example:

    "C:\Program Files\Java\jdk-21\bin\java.exe" --enable-preview --enable-native-access=ALL-UNNAMED -jar directml-workbench-all.jar

If your Java 21 installation is somewhere else, adjust the path accordingly.
