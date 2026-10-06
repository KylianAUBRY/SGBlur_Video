# Privacy and GDPR

!!! note "Planned"
    Completed in step 9 from the implemented behaviour. The retention rules
    are already designed in [Architecture § Data retention](../design/architecture.md#data-retention-privacy).

## Outline

1. Principle: recall over precision
2. What is stored, where, and for how long
3. `keep=1`: encrypted regions and their lifetime
4. What is never logged
5. Irreversibility of the blur ([ADR-0004](../adr/0004-irreversible-blur.md))
6. Known limits (detector misses, tiny faces, night, unusual plates)
7. Reporting a privacy leak ([SECURITY.md](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/SECURITY.md))
