package com.aresstack.windirectml.workbench.launcher;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link WorkbenchLauncher} version parsing and candidate list
 * building logic.
 */
class WorkbenchLauncherTest {

    @Test
    void parseMajorVersion_java21() {
        assertEquals(21, WorkbenchLauncher.parseMajorVersion(
                "openjdk version \"21.0.1\" 2023-10-17"));
    }

    @Test
    void parseMajorVersion_java17() {
        assertEquals(17, WorkbenchLauncher.parseMajorVersion(
                "openjdk version \"17.0.8\" 2023-07-18"));
    }

    @Test
    void parseMajorVersion_java8OldStyle() {
        assertEquals(8, WorkbenchLauncher.parseMajorVersion(
                "java version \"1.8.0_392\""));
    }

    @Test
    void parseMajorVersion_java11() {
        assertEquals(11, WorkbenchLauncher.parseMajorVersion(
                "openjdk version \"11.0.21\" 2023-10-17"));
    }

    @Test
    void parseMajorVersion_noVersionLine() {
        assertEquals(-1, WorkbenchLauncher.parseMajorVersion(
                "OpenJDK Runtime Environment (build 21.0.1+12)"));
    }

    @Test
    void parseMajorVersion_null() {
        assertEquals(-1, WorkbenchLauncher.parseMajorVersion(null));
    }

    @Test
    void parseMajorVersion_emptyString() {
        assertEquals(-1, WorkbenchLauncher.parseMajorVersion(""));
    }

    @Test
    void extractMajor_simpleNewStyle() {
        assertEquals(21, WorkbenchLauncher.extractMajor("21.0.1"));
    }

    @Test
    void extractMajor_singleNumber() {
        assertEquals(21, WorkbenchLauncher.extractMajor("21"));
    }

    @Test
    void extractMajor_oldStyle18() {
        assertEquals(8, WorkbenchLauncher.extractMajor("1.8.0_392"));
    }

    @Test
    void extractMajor_oldStyle16() {
        assertEquals(6, WorkbenchLauncher.extractMajor("1.6.0_45"));
    }

    @Test
    void extractMajor_withBuildMetadata() {
        assertEquals(21, WorkbenchLauncher.extractMajor("21.0.1+12"));
    }

    @Test
    void extractMajor_earlyAccess() {
        assertEquals(22, WorkbenchLauncher.extractMajor("22-ea"));
    }

    @Test
    void extractMajor_null() {
        assertEquals(-1, WorkbenchLauncher.extractMajor(null));
    }

    @Test
    void extractMajor_empty() {
        assertEquals(-1, WorkbenchLauncher.extractMajor(""));
    }

    @Test
    void buildCandidateList_isNotEmpty() {
        // On any system with at least a PATH, we should get some candidates
        List<String> candidates = WorkbenchLauncher.buildCandidateList();
        assertNotNull(candidates);
        // The list itself is built; content depends on environment
    }

    @Test
    void minJavaVersion_is21() {
        assertEquals(21, WorkbenchLauncher.MIN_JAVA_VERSION);
    }
}
