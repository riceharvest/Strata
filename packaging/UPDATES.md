# Updates: the signed feed, the component updater, the app updater

This page is for developers who build, sign and publish releases. Users update with [INSTALL.md](../docs/INSTALL.md#updating).
It covers the three updated parts, the feed format, what a PC checks, and the release steps.

Everything that downloads software on a user's PC sits behind one explicit step: `strata update` in the Terminal, or
"Update engine" or "Update Strata" in the app's Settings. Starting a model and serving never download anything
(`STRATA_NO_ACQUIRE` stays on for them; the update step is a separate Control process that is allowed to). Engines are
separate from the app and ship more often. An app-only update downloads no engine bytes, because the Windows and Linux installers are built
without the engine. An engine update downloads only the package for that PC's GPU.
**Models and saved tuning presets are never downloaded, replaced or deleted by any update.**

## The three things that are updated

| What | How | Signed with |
|---|---|---|
| Engine package (per platform / backend / variant) | Control's component updater: download with resume, verify, install beside the old one, test-start, switch | the feed key (Ed25519) |
| Control (Python runtime, `strata_control`, CLI, server) | the same updater: staged beside the running one, run once, then `components/control/active.json` points at it | the feed key |
| The desktop app | Tauri's signed updater in the shell, reached by the host through a token-protected loopback bridge | the Tauri updater key (minisign) |

Two separate private keys, both kept by the owner, neither in git, neither the same as Apple/Windows code signing:

* **feed key**: Ed25519, signs `strata-feed.json` and every engine package manifest.
* **Tauri updater key**: minisign (what `tauri signer generate` makes), signs the app installers.

<a id="the-feed"></a>
## The feed (`strata-feed.json` + `strata-feed.json.sig`)

`strata-feed.json.sig` is `{"alg": "ed25519", "key_id": "<16 hex>", "value": "<base64>"}` over the **exact bytes** of
`strata-feed.json` (no canonicalisation). The annotated example below illustrates the format; it is not a current signed feed.
Comments are for this document only; actual feed files must be strict JSON.

```jsonc
{
  "schema_version": 1, "feed": "strata-updates", "channel": "stable",
  "sequence": 12,                           // goes up with every feed; an older one than the last accepted is refused
  "generated_at": "2026-10-20T10:00:00Z", "expires_at": "2027-10-20T10:00:00Z",   // expires_at may be null
  "app": {                                  // exactly Tauri's static updater JSON, plus sha256/size per platform
    "version": "0.2.0", "notes": "...", "pub_date": "2026-10-20T10:00:00Z",
    "platforms": {"linux-x86_64": {"url": "app/Strata-0.2.0-linux-update.AppImage", "signature": "<minisign, base64>", "sha256": "...", "size": 71000000}}
  },
  "control": {"version": "0.1.50", "packages": [{"platform": "linux-x86_64", "format": "tar.gz", "url": "control/...tar.gz",
                                                  "size": 0, "sha256": "...", "installed_bytes": 0, "entry": "strata.py"}]},
  "engine": {"packages": [{
    "version": "0.1.48", "platform": "linux-x86_64", "backend": "sycl",
    "archs": ["intel-arc"], "min_driver": {}, "min_runtime": null,
    "build_id": "be885026f96a-oneapi2026.1", "format": "tar.gz", "url": "engine/strata-engine-0.1.48-linux-x86_64-sycl.tar.gz",
    "size": 505568970,                      // bytes of the archive
    "sha256": "...",                        // of the archive
    "manifest_sha256": "...",               // of the package's engine-component.json (it holds every file's hash)
    "installed_bytes": 840893419, "notes": ""}]},
  "dsv41": {"packages": []}                 // optional: the separate DeepSeek engine packages, same entry shape as "engine"
}
```

The DeepSeek V4.1-Flash engine `strata-dsv41` reaches a PC in one of two ways. The engine package carries it beside `strata`
(`package_engine.py --require-dsv41`), or it is a separate package in the feed's `dsv41` member, packed with
`pack-engine --prefix strata-dsv41`. The separate package installs in `<data>/engine-dsv41/` beside the engine, never inside it
(`strata_control/dsv41_engine.py`), and is fetched when DeepSeek is chosen. `STRATA_DSV41_EXE` points setup at a build of your own.

`url` is relative to the feed's own location (or an absolute `https://` address; a loopback `http://` is allowed so a local test
feed works; a local feed folder may only name files inside itself).
For a GitHub release the assets are flat: pack with `--url-prefix ""` and upload every file beside the feed.

### What a PC checks, in order

1. the signature of the feed (key ids it trusts: `strata_control/update_keys.json`, plus `STRATA_UPDATE_TRUSTED_KEYS` for a test setup, which
   the CLI announces);
2. `channel`, `expires_at`, and `sequence` not lower than the last accepted feed (the downgrade guard for a replayed old feed);
3. the engine package that fits **this** PC (`feed.select_engine`): platform, backend (SYCL for an Intel Arc card), `min_driver`,
   `min_runtime` against Control; newest version wins, and equal versions prefer the build that needs the newer driver;
4. never older than what is installed (an update never downgrades; only `strata components rollback` goes back).

### What an engine update does

download (resume, progress) -> SHA-256 of the archive = the feed's -> unpack into `components/.s/<4 hex>/` (short on purpose: Windows paths are limited to 259 characters unless long paths are on; files are written through extended-length paths) (names and links checked,
total size limited to the feed's `installed_bytes`) -> the manifest is the one the feed names (`manifest_sha256`) and says what the feed says
-> the manifest's own signature verifies -> every file's hash -> fits this PC -> **test-start from the staging folder through the real start
path** (isolated owner runtime and profile, `component:strata-engine`, an installed model, one-token probe, stop) -> only then rename
into `components/engine/` and re-resolve to make sure a start now picks it. Any failure leaves `engine/` exactly as it was.
The previous version is kept (`state.json` `engine.previous`); the one before it is removed. With no model installed yet (first run)
the test is `--version` only, and the report says so.

Not while an engine is serving: the check uses the activity lease `strata serve` and the owner runtime already hold. `--wait SECONDS` waits;
a terminal asks whether to wait; scripts get exit 7. The download itself may run first; the check is repeated right before the test start and
before the switch.

Saved tuning presets: when the engine identity changed (version, build id, manifest hash, backend) `tuning/recheck.json` gets a
`pending` entry (`engine_changed`, from/to identities). Presets are not opened. The tuner offers the ~2 minute check for them and calls
`preset_marks.clear(id)`; rolling back to the engine they were measured on clears the debt again.

### Commands

    strata update [--check] [--app] [--no-engine] [--no-control] [--feed URL|DIR] [--wait S] [--force] [--model-config RUN.json]
    strata components update            # the same
    strata components rollback          # back to the previous engine; the next update will not re-offer the version rolled back from (--force does)
    strata components engine            # shows "update available (...)" from the last check, no network

Exit codes: 0 done or nothing to do, 5 refused (signature, hash, downgrade, no trusted key, acquiring forbidden),
6 failed with the old version still in use (network, test start), 7 an engine is serving, 130 stopped.
`--output json` / `--output jsonl` (progress events, last line is the result) are what the app's host uses.

## How the owner makes the real keys (once)

On an offline machine, outside any git checkout:

    python packaging/feed_tool.py keygen --out D:\strata-keys          # feed-key.json (SECRET) + feed-key.pub.json
    npx tauri signer generate -w D:\strata-keys\tauri-updater.key      # asks for a password; prints the public key

Back both up (a lost feed key means every installed PC needs a build with a new key; a lost Tauri key means the same for the app).
Then, in git, once, in the commit that ships the first signed release (the release key id is `5d1bc23abbeffd0d`):

* the `keys` entry from `feed-key.pub.json` into `strata_control/update_keys.json`;
* the Tauri public key into `apps/desktop/src-tauri/tauri.conf.json` (`plugins.updater.pubkey`) and into `update_keys.json` as `app_minisign_public`
  (the Terminal verifies the app installer with it).

Only release keys live in `update_keys.json`. A tester or local-test key is never added there: it is passed with `STRATA_UPDATE_TRUSTED_KEYS`
(a file of extra public keys; `packaging/make_local_feed.py` writes one as `test-trust.json`), so private tester builds keep working
without their key being part of a release build. A build whose `update_keys.json` is empty (a source checkout from before the keys) refuses every
feed with "this build trusts no update key".
A key rotation is a release that carries both public keys; the old private key signs that release, the new one signs the next.

## Releasing

Keys live only on the release PC (`feed-key.json`, `tauri-updater.key`; the Tauri key has no password). Every build box builds WITHOUT a key:
the default `tauri.conf.json` has `createUpdaterArtifacts: false`, so `tauri build` needs no signing key and makes the same installer bytes.
All signing happens on the release PC, on the finished files.

**What is built (per OS)**

* *Full installers* (the first download, engine inside): the Linux AppImage with the Intel Arc (SYCL) engine. It carries ONE signed
  engine package in `engine-bundle/`.
* *Update installers* (the only files `latest.json` names): built with `--config src-tauri/tauri.update.conf.json`, which drops `engine-bundle`:
  the base AppImage before the engine is repacked into it (`Strata-0.2.3-linux-update.AppImage`).
  Why: an app update must not carry an engine. Engines come from the signed feed (`feed.select_engine`; `tests/terminal/test_release_matrix.py`).

**Steps**

    python packaging/feed_tool.py sign-package <engine package folder> --key <feed-key.json>        # every engine package
    python packaging/feed_tool.py pack-engine  <engine package folder> --out RELEASE/engine --url-prefix=
    python packaging/feed_tool.py pack-control <control folder> --version 0.1.48 --platform P --out RELEASE/control --url-prefix=   # --platform linux-x86_64
    # The Control version is the VERSION file (0.1.48), not the app version: a Control package newer than the installed one is
    # installed by `strata update`.
    # tauri build --bundles appimage (no engine) is the update AppImage; repack-appimage.sh puts an engine package into a copy for the full AppImage
    # on the release PC, for the update .AppImage:
    npx tauri signer sign -f <path to tauri-updater.key> <artifact>                                 # writes <artifact>.sig
    python packaging/feed_tool.py set-app-platform --app-json RELEASE/tauri-latest.json --target linux-x86_64 --version 0.2.3 --archive <update .AppImage> --sig <update .AppImage.sig>
    python packaging/feed_tool.py build-feed --out RELEASE --sequence N --key <feed-key.json> --engine RELEASE/engine --control RELEASE/control --app-json RELEASE/tauri-latest.json
    python packaging/feed_tool.py verify-feed RELEASE --pub <feed-key.pub.json>

`set-app-platform` checks each `.sig` against the public key in `tauri.conf.json` over the exact bytes, so a file signed with another key, or changed after
signing, stops the release. The Mac dmg is not in `latest.json` (the updater uses `Strata.app.tar.gz`) and needs no Tauri signature; it is downloaded by hand.

The Linux AppImage needs no extra step: its `.sig` file from `tauri build` is the one to pass to `set-app-platform`.

`build-feed` also writes `RELEASE/install-manifest.txt`, the file the one-line installer reads (`install.sh`): one line per Control package,
`platform version url sha256 size`, taken from the `*.entry.json` files in `RELEASE/control`. The Control package is the Terminal that `install.sh` unpacks (it must hold `bin/strata`).
`install.sh` unpacks with `tar -xzf`, so pack Control as `tar.gz` (`build-feed` leaves a pack in the wrong format out of the manifest and says so).
`install.sh` knows `linux-x86_64`; a platform with no line gets "this release has no Terminal for <platform>". Upload `install.sh` from `packaging/`
unchanged: its built-in address is the release folder.

**Linux ARM64 (DGX Spark) has its own feed file.** A Control up to 0.1.52 refuses a feed that names a platform it does not know (`the feed's control entry names an unknown platform`), so the `linux-aarch64` entries are not in `strata-feed.json`, which every released client reads. They are in `strata-feed-linux-aarch64.json` (+ `.sig`, same format, same key, its own `sequence`, aarch64 entries only), uploaded to the same release. A Linux ARM64 Terminal reads that file by default, from a folder or a release address (`strata_control.feed.feed_name()`); a source that names its own `.json` file is used as given. Until every client skips unknown platforms, each release must publish both files, and the aarch64 file's sequence must never go down.

The feed is hosted as the files of a GitHub release of `Niko1221/Strata`. The built-in address is
`https://github.com/Niko1221/Strata/releases/latest/download` (`strata_cli/update_cmd.py` `DEFAULT_FEED` adds `/strata-feed.json`; `install.sh`
uses the folder). GitHub assets are flat, so build the packages with `--url-prefix ""` and upload every file of the release folder
as an asset: `strata-feed.json`, `strata-feed.json.sig`, `install-manifest.txt`, `install.sh`, the engine and Control archives and the
app installer with `latest.json` (the Tauri updater endpoint). `STRATA_FEED` (installers) and `--feed` / `STRATA_UPDATE_FEED` (`strata update`) override the
address. Nothing in this tree uploads anything.

Publish as a normal release (latest): a pre-release is not found by `latest/download`, so the installers, `strata update` and the app updater would get a 404. The `latest` release must carry every file listed above, so a later hotfix release repeats them all. The app updater reads
`https://github.com/Niko1221/Strata/releases/latest/download/latest.json`.

## Testing locally

Everything above runs against a folder or a loopback server: `tests/terminal/test_update_feed.py` (signatures, hashes, downgrades, selection),
`tests/terminal/test_updater.py` (end to end against a loopback HTTP feed with stub engines: update, test-start, switch; a failing test start
keeps the old engine; an interrupted download resumes; rollback; Control; the app installer's minisign check), `apps/host/tests/component-updates.test.mjs`
(the Settings buttons). For a real engine on a box, build a package with `packaging/package_engine.py`, serve it from a local feed made by `packaging/make_local_feed.py`, and run `strata update --feed` against it.

## History / older measurements

- **Keys:** the feed key and the Tauri updater key were created once and the feed key's public half was committed in `strata_control/update_keys.json`
  with the first signed release (0.2.0-beta). Terminals from before that build trust no key and refuse every feed.
- **Engine-less installers:** an app update once carried an engine. The Windows and Linux installers now drop `engine-bundle`
  (`tauri.update.conf.json`), so one `latest.json` entry per OS serves every card variant.
