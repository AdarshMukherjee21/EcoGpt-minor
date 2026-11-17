# ecoGPT Chat UI (vendored deepseek-harness) — runbook

Status: wiring complete on the proxy side (`proxy/chat_ui.py`), **not yet mounted**
into `proxy/app.py`. Mounting is held until the concurrent app.py edits land.

## What this is

The user-facing chat UI is the vendored `deepseek-harness` web app:

- Frontend+backend composition: `vendor/dsh-harness/` (original `deepseek-harness/` checkout stays pristine)
- Served by the harness's own Node host on the private loopback port **3180**
- The daemon reverse-proxies it at **/chat** (prefix-stripping; document-relative
  client calls resolve to `/chat/api/...` automatically)
- All model traffic is pinned to the ecoGPT gateway (`POST /v1/messages` on
  127.0.0.1:18090) — the sidecar holds no upstream credentials of its own

## Version pin + update procedure

The sidecar runs the **pinned npm distribution** (no source checkout needed at
runtime):

```sh
npx -y @deepseek-ai/dsh@0.2.2-rc.3 --profile web --no-open --port 3180
```

- `DSH_NPM_PIN` (`proxy/chat_ui.py`) and `dsh_npm_pin` (`ChatUIConfig`) carry the
  pin; bump both together.
- Update procedure:
  1. Check the release notes at the deepseek-harness repository (developer
     preview; compatibility-breaking changes are common).
  2. `npm view @deepseek-ai/dsh versions` to pick the next pin.
  3. Re-verify with the Phase-0 spike: boot the sidecar, confirm the entry URL
     prints, a chat turn appears in the dashboard ledger, and the model picker
     lists exactly the gateway's exposed models.
  4. Update `DSH_NPM_PIN` in `proxy/chat_ui.py`, run `pytest tests/test_chat_ui.py`.

### Update procedure (vendored copy branding)

Source-built features (rebranding, sidebar Dashboard button) require a source
build of `vendor/dsh-harness/` (pnpm ≥11, Node ≥24.9 to build). Those patches
are documented in the vendored tree:

- `packages/client/ui-primitives/src/FishLogo.tsx` — brand mark (leaf)
- `packages/client/ui-primitives/src/BrandWordmark.tsx` — brand wordmark (text)
- `packages/client/locale/src/locales/*.ts` — `brand.localBuild` → `ecoGPT`
- `packages/client/ui-brand-official/src/client/index.ts` — PixelWhale hero frame unmounted
- `packages/client/ui-conversation/.../EmptyHero.tsx` — SMIL swim morphs removed
- `apps/web/index.html` + `apps/web/public/manifest.webmanifest` — title + PWA names
- Regenerate PNG icons with `./.venv/Scripts/python.exe tools/make_ecogpt_icons.py`

## Model restriction (config-only)

- `llm-deepseek.baseURL = http://127.0.0.1:18090` in the dsh user profile
  (`~/.dsh/profiles/web/cordis.patch.yml`) → every chat request flows through
  the ecoGPT gateway pipeline (compression → router → tracing → provider).
- The gateway performs **credential substitution**: the sidecar's dummy
  `DEEPSEEK_API_KEY=ecogpt-local-dummy` is never honored upstream.
- The model picker is seeded from the gateway's exposed models; a live listing
  upgrade path exists via `GET /models` on the gateway (still needs the alias
  route + row enrichment in `proxy/app.py` — held wiring).
