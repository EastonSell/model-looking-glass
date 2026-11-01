# Offline viewer dependencies

These are unmodified assets extracted from the pre-existing pinned npm archives supplied in the task workspace. No package was installed or executed. On 2026-11-01 each archive's SHA-512 was compared successfully against `dist.integrity` returned by the official registry's exact-version metadata endpoint. `provenance.json` records those primary-source URLs, matched archive integrity values, and SHA-256 for every extracted asset (including licenses).

- Three.js **0.128.0**, classic browser build and matching classic OrbitControls. MIT license: `THREE-LICENSE.txt`.
- @fontsource/inter **5.3.0**, Latin normal 400 and 600. SIL Open Font License: `INTER-LICENSE.txt`.
- @fontsource/jetbrains-mono **5.3.0**, Latin normal 400 and 600. SIL Open Font License: `JETBRAINS-MONO-LICENSE.txt`.

All complete upstream license texts are preserved. The app loads only local assets; provenance URLs document verification, not runtime dependencies. Three r128 is intentionally pinned to the supplied known classic-script build; this is not a claim that it is the latest release.
