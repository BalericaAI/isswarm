You have access to runbook.md, the authoritative runbook for the
Guardian Telemetry Swarm.

Task: compile the Guardian application.

1. Ensure the Guardian source is present by cloning
   https://github.com/BalericaAI/isswarm and working in the demo01
   directory (the directory containing runbook.md and main.py).

2. Follow Section 0 of runbook.md ("Compile Protocol
   (Machine-Executable)") exactly: execute STEP 1 through STEP 6 in
   order, comparing each output to the expected marker.

3. Do not skip steps, do not ask for API keys or credentials
   (compilation requires none per Section 0.1), and stop at the first
   failing step.

4. Report the outcome using only the Section 0.3 result contract:
   COMPILE_RESULT: PASS, or COMPILE_RESULT: FAIL with FAILED_STEP and
   EVIDENCE.

Do not run the security swarm, do not configure .env, and do not go
beyond compilation (Section 0.4).
