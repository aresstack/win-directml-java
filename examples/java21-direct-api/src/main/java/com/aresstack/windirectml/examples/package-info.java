/**
 * Demonstrates direct Java 21 in-process use of the local ML runtime
 * <b>without</b> the JSON-RPC sidecar.
 *
 * <h2>Prerequisites</h2>
 * <ul>
 *   <li>Model weights at {@code model/all-MiniLM-L6-v2/} (run
 *       {@code scripts/download-minilm.ps1} first).</li>
 *   <li>Java 21+.</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * ./gradlew :examples:java21-direct-api:run
 * }</pre>
 */
package com.aresstack.windirectml.examples;
