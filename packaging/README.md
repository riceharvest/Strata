# Strata packaging: targets, recipes and locks

This page is for developers who build the desktop app, the Terminal archive and the engine packages. To install Strata, use
[docs/INSTALL.md](../docs/INSTALL.md). It describes the current release (app and Terminal 0.2.5, Control and engines 0.1.50).
How packages are signed and published is in [UPDATES.md](UPDATES.md), and the package format is in [ENGINE-COMPONENT.md](ENGINE-COMPONENT.md).

Release assets show what was published; the platform guides and retained reports show what was tested. A published package does not
qualify every GPU, model or feature on its platform.

## Quick start

| I want to | See |
| --- | --- |
| Build the desktop app | [Desktop app](#desktop-app) |
| Build an engine package | [ENGINE-COMPONENT.md](ENGINE-COMPONENT.md#building-a-package) |
| Build the Terminal archive | [The Terminal archive](#the-terminal-archive) |
| Sign and publish a release | [UPDATES.md](UPDATES.md#releasing) |
| Test an update against a local feed | [UPDATES.md](UPDATES.md#testing-locally) |

## The target matrix

A backend name alone is not a target. Each row is an OS x CPU ABI x backend tuple, and a row is
`qualified` only after the acceptance checklist has actually run on it. This fork targets one machine, an Intel Arc Pro B70 on
Linux, so the NVIDIA (CUDA), AMD (HIP), Apple (Metal) and Windows/macOS rows and their recipes are gone.

| Target | App artifact | Engine package recipe | Status |
| --- | --- | --- | --- |
| Terminal `linux-x86_64` | self-contained `.tar.gz` | any | the only Terminal target (`package_terminal.py`). |
| `linux-x86_64-sycl` | | `packaging/build_sycl_linux.sh` (one JIT package for Alchemist and Battlemage) | **engine package built; tested on an Arc A750** (Intel Arc is an official platform since 0.1.40.2, docs/INTEL_ARC.md). Arc B-series (Battlemage) cards are supported and untested by us: no B-series card was available for the package (the B70 tests of 2026-10 were with the source build), and the Terminal says so once and tells the person how to report a problem. `sycl/setup_intel.py` (Docker) remains the source-build route. |

**Cross-compilation is not attempted for the desktop app.** Tauri bundles are per-OS. What *is* portable is
everything below the carrier: the Python contracts, the owner transport, the checks, and the engine
build recipes.

## The private Node runtime

The desktop bundle ships a pinned standalone Node beside the App Host, so an end user supplies no runtime. It is not committed
(about 117 MB, an input rather than source). Fetch and verify it:

```sh
apps/desktop/scripts/fetch-node.sh                 # host platform
apps/desktop/scripts/fetch-node.sh --target linux-x64
apps/desktop/scripts/fetch-node.sh --check         # verify what is installed
```

The script refuses an archive whose SHA-256 differs from the digest recorded for that target. The digests come from nodejs.org's
`SHASUMS256.txt` for the pinned version. To bump the version, change `STRATA_NODE_VERSION`, add the new digests and run `--check`.
`fetch-node.ps1` is the Windows equivalent.

## Desktop app

```sh
cd apps/desktop
./scripts/fetch-node.sh                                                      # the sidecar (once)
npm install --prefix ../host && npm --prefix ../host run build                # the UI

# Linux (on Linux): AppImage plus a deb
cargo tauri build --bundles appimage deb
```

Uninstall retention (checklist P10) is a property of *where
the app keeps its data*, not of an installer flag: models, profiles and sessions live outside the
install directory (`~/.strata` and the platform state directories), so removing the application
cannot remove them.

## Strata Control and the engine in the installed app

The installer carries Control, and on a Mac the Metal engine too. Neither fetches anything when it runs.

**Control** (`<install>/control`, built by `apps/desktop/scripts/stage-control.py`) is the same bundle as the Terminal archive:
the pinned CPython 3.12.15, the hash-locked wheels, `strata_control`, `strata_cli` and `serve`, plus a `BUNDLE.json` with the
interpreter and lock digests. The host finds it beside itself or through `STRATA_CONTROL_DIR`, runs it on its own interpreter
(`PYTHONNOUSERSITE=1`, no `PYTHONHOME` or `PYTHONPATH`) and starts a model with `strata profile create|use-config` and
`strata runtime start`. The engine then runs under the owner runtime (loopback TCP and a 256-bit token on Windows, a private
socket elsewhere; a Job Object or process group; leases), and a second client such as the Terminal can attach to the same instance.

The app starts each engine with `runtime start --die-with-pid <host>`, so the engine ends with the host. The host ends with the shell
(a kill-on-close Job Object on Windows; the host watches its parent everywhere). The app is a single instance: a second launch focuses
the open window. One app can run several engines (profiles `app-1`, `app-2`, ...). Every close path is in [docs/CLOSING.md](../docs/CLOSING.md).

**The engine** comes with the installer (`<install>/engine-bundle/<package folder>`, the same layout as below); the first start checks it against its own manifest and copies it into the per-user component folder, where Control finds it like any engine that `strata update` installed. It is a component in the per-user component folder
(`$XDG_DATA_HOME/strata/components`, or `STRATA_COMPONENTS_DIR`):

    components/engine/<version>-<platform>-<backend>/
        engine-component.json     schema 2: id/name "strata-engine", version, platform, backend, archs, min_driver,
                                  build_id, min_runtime (optional), entrypoint, files [{path, sha256, bytes}],
                                  signature (null for a local package, set by the release feed); full table and the checks in ENGINE-COMPONENT.md
        strata  lib*.so*          the binary and the GPU runtime libraries beside it

A run config names it as `"exe": "component:strata-engine"`; at start `strata_control/engine_component.py` picks the package
for this PC (platform, then the backend its GPU needs, or `STRATA_ENGINE_BACKEND`; `min_driver` and `min_runtime` are checked; the newest version wins), checks every file's size and
SHA-256 (a start hashes the whole package; a mismatch is `EngineIntegrityError` and nothing starts) and puts the package
folder first on the library path.
Nothing matching is `EngineMissing`: `strata serve` ends with exit 5 and "No engine for this PC was found in ... The app never downloads an engine
while it starts or serves", and no fetch. That is the start itself. Everything before it is a warning, not a refusal: `strata launch` says in plain
words that it could not find an engine for this computer, asks whether to go on (Enter means yes), downloads the model and ends with the next step
(exit code 0, nothing started), and the app shows the same warning with an "Update engine" button in Settings. `packaging/package_engine.py` makes a local package from a build folder.

When no package matches, `strata serve` ends with exit 5 and "No engine for this PC was found". It never fetches. Before that point
everything is a warning: `strata launch` says it could not find an engine, asks whether to go on, downloads the model and ends with
the next step (exit 0, nothing started). The app shows the same warning with an "Update engine" button in Settings.
`packaging/package_engine.py` makes a local package from a build folder.

**No acquisition at run time.** `STRATA_NO_ACQUIRE=1` (set by the Tauri shell for the host, and by the host for Control) makes
`strata_control/acquisition.py` refuse pip, downloads and compiles. On the host side `server/acquisition.mjs` refuses the llama.cpp
and cloudflared fetches (a connector placed in `components/cloudflared/` is used as it is). Models the user chooses to download are a
separate, explicit action. `apps/host/tests/no-acquire.test.mjs` fails on any outbound request or installer process.
`apps/desktop/scripts/installed-flow.mjs` drives an installed app end to end.

## The Terminal archive

The Terminal bundles a private Python and its dependencies for `windows-x86_64`, `linux-x86_64`, `linux-aarch64` (DGX Spark) and `darwin-arm64`
(`package_terminal.py --target ...`, built on the target OS).

- The interpreter is python-build-standalone CPython 3.12.15, digest-checked at build time.
- Packages come from `packaging/locks/runtime-<target>-cp312.txt` and are installed with
  `pip install --require-hashes --no-deps --only-binary=:all:` at build time only. `lock_wheels.py` makes the lock; review its diff when a pin changes.
- Wrappers: `bin/strata.cmd` and `bin/strata.ps1` on Windows, `bin/strata` on POSIX. `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`, `setup.sh`
  and `update.sh` use a `python/` folder beside them when there is one.
- `smoke_terminal.py <archive>` extracts an archive into a clean environment (a path with a space, a PATH with no Python) and runs
  `--version`, `doctor`, `profile list`, `install` and `serve --engine mock`.
- Check each archive's `BUNDLE.json` and lock digests. An old archive is not upgraded when the source lock changes.

## What a platform must produce before it is qualified

1. The app launches and the window serves the supplied UI from the host's origin.
2. The owner transport works: one owner per installation, a second client attaches, a stale owner can be recovered, and no client can stop a process it does not own.
3. Real inference on a supported pack, with the engine build hash and the model hashes recorded.
4. `docs/work/QUALIFICATION-KIT.md` run end to end, with raw evidence kept.

## Release checks and gaps

* There is no CI. Every recipe here is written to be run by hand; making them automated is separate
  work.
* Signing needs the owner's credentials, which are never in this tree. The update feed and every engine package are signed with the feed key
  (`feed_tool.py`, [UPDATES.md](UPDATES.md)); the app installers with the Tauri updater key.
* No font or asset redistribution evidence has been gathered (D05). The app currently ships the
  brand assets from the supplied UI ZIP, which is fine for an internal candidate and not sufficient
  for redistribution.
* The terminal distribution bundles a private Python runtime and its dependencies for `linux-x86_64`
  (`package_terminal.py`). The interpreter is
  the pinned python-build-standalone CPython 3.12.15 (digest-checked at build time); the packages come from
  `packaging/locks/runtime-<target>-cp312.txt`, installed with `pip install --require-hashes --no-deps
  --only-binary=:all:` at BUILD time only (`lock_wheels.py` makes the lock; review its diff when a pin changes).
  Wrapper: `bin/strata`; setup.sh and update.sh use a `python/` folder beside them when there is one.
  `smoke_terminal.py <archive>` extracts an archive into a clean environment (a path with a space, a PATH with no
  Python) and runs `--version`, `doctor`, `profile list`, `install` and `serve --engine mock`.
* `.gitignore` coverage for build output was added late. `apps/desktop/src-tauri/target` and the
  Node sidecar were briefly committed, which left ~324 MB in the git object store. The files are
  untracked now and the ignore rules are in place, but the history is not rewritten: that is a
  deliberate choice, since rewriting published history is worse than a large local repository. A
  developer cloning this branch gets the current tree, and the sidecar arrives via
  `fetch-node.sh`.
