These seven Python files are the exact source bytes from commit
`02e1c5d5c79813e3d67bbd0468b599926b4b9f83`, retained solely as offline test
fixtures for the source-first allocation and transition callers. The callers
pin that published revision; the live package can evolve independently. Tests
compare each fixture's SHA-256 to the caller's own pins before executing it in
a fresh child interpreter.
