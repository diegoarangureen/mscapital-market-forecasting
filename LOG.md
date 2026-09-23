
## 2026-09-23 23:02 CEST — Credential rotation (security hygiene)
- Rotated both automation credentials after they appeared in scheduled-task prompts (treated as exposed).
- Kaggle API token: new token generated and installed; old token expired via account settings and verified dead (401).
- GitHub: migrated from a classic PAT to a fine-grained PAT scoped to this repo only (Contents read/write, Metadata read-only, 30-day expiry). Push verified working. Old classic PAT revoked (see note below).
- Recovery procedure for sandbox wipes now uses vault-based browser sign-in to generate fresh credentials; no persistent secret is stored in any prompt, message, or file outside the local git config.
