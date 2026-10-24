# Engine component manifest (schema 2)

The format of the engine package (this fork builds one: the Linux Intel Arc / SYCL engine, for the Arc Pro B70). Control (`strata_control/engine_component.py`) reads it, checks it, and only then starts the engine.
`packaging/package_engine.py` writes it. A package you make locally is unsigned and loads. A package in the release feed is signed with the
feed key (an Ed25519 key the owner keeps offline), and `strata update` / "Update engine" install it only when that signature verifies
([UPDATES.md](UPDATES.md)). The installers carry an engine of their own, so a new install needs no download to start a model.

## Folder

    <components>/engine/<version>-<platform>-<backend>[-<variant>]/
        engine-component.json
        strata                           the entrypoint
        lib*.so*                         the GPU runtime libraries it loads from beside itself
        BUILD.json  LICENSE*             optional records
        strata-vision  strata-dsv41 ...  anything else the build needs, listed in the manifest like the rest

`<components>` is `$XDG_DATA_HOME/strata/components` or `STRATA_COMPONENTS_DIR`.
A package folder is immutable: a new build is a new folder.

## `engine-component.json`

| field | type | meaning |
|---|---|---|
| `schema_version` | 2 | schema 1 packages (no `archs`, `min_driver`, `build_id`, `signature`; `cuda_archs` instead of `archs`) still load |
| `id`, `name` | `"strata-engine"` | the component name; anything else is refused |
| `version` | string | the engine release, e.g. `0.1.45`; the newest compatible version wins |
| `platform` | string | `linux-x86_64` (must match this PC) |
| `backend` | string | `cuda`, `hip`, `sycl`, `metal`, `cpu`; must be a combination `BACKEND_MATRIX` commits to, and the one this PC's GPU needs |
| `archs` | list of strings | what the build contains. SYCL: a descriptive tag (`intel-arc`, a JIT build) or the AOT device names (`bmg-g31`). Required for every non-`cpu` backend; recorded |
| `min_driver` | `{vendor: version}` | the oldest driver the build runs on, e.g. `{"nvidia": "580.65"}`. Enforced for NVIDIA where `nvidia-smi` answers; an unreadable driver is not a refusal |
| `build_id` | string | which engine build this is: source commit and toolkit, e.g. `be885026f96a-oneapi2026.1`. Required |
| `variant` | string, optional | tag when one backend has several builds on a platform; goes in the folder name. With equal versions the build that needs the newer driver is preferred when both fit the card |
| `min_runtime` | string, optional | the oldest Control that may start it |
| `entrypoint` | string | relative path of the binary; must be in `files` |
| `files` | list | `{path, sha256, bytes}` for **every** file of the package except the manifest; paths are plain relative POSIX paths that stay inside the folder; no duplicates |
| `notes` | string, optional | free text |
| `signature` | `null` or `{alg, key_id, value}` | see below |

Example (shortened):

```json
{"schema_version": 2, "id": "strata-engine", "name": "strata-engine", "version": "0.1.45",
 "platform": "linux-x86_64", "backend": "sycl", "archs": ["intel-arc"], "min_driver": {},
 "build_id": "be885026f96a-oneapi2026.1", "entrypoint": "strata",
 "files": [{"path": "strata", "sha256": "<64 hex>", "bytes": 191000000}], "signature": null}
```

## What Control checks before a start

In this order, each refusal names the cause; none of them starts anything and none of them downloads anything:

1. the manifest parses strictly (types, schema, paths, archs, driver, signature shape);
2. the platform and backend are this PC's;
3. `min_runtime`, CUDA archs vs. the card, `min_driver` vs. the driver;
4. the **signature policy** (below);
5. **every file** exists, is a regular file inside the folder (no symlinks), has the listed size, **and matches its SHA-256**
   (a start hashes the whole package; the listing commands `strata components engine` hash only the entrypoint unless given `--full`);
6. on POSIX the entrypoint is executable.

If a package fails (5) or (4) and no other package fits this PC, the start ends with `EngineIntegrityError`
(exit code 5): "The engine package failed its integrity check, so it was not started: 0.1.45 linux-x86_64-sycl:
libsycl.so.8 does not match its recorded SHA-256. Repair or reinstall the engine component with Update engine in the app's
Settings or `strata update` in the Terminal". A package
for another GPU or an older driver is `EngineMissing` with its reason. A tampered newer package never falls back silently
to nothing: a good older package is used and the tampered one is listed under `problems` in `strata components engine`.

## Signature

`signature` is `null` in a package made locally. The release feed's packages carry one; the updater registers the verifier
(`strata_control/signing.py`) and requires it for every package it installs:

    engine_component.register_signature_verifier(fn)       # fn(payload: bytes, signature: dict, component) -> None, raises if bad
    engine_component.signing_payload(manifest_dict)         # canonical bytes the signature covers

The payload is the manifest without its `signature` member, as canonical JSON (sorted keys, `,` and `:` separators,
UTF-8). It contains every file hash, so one signature covers the whole package. Policy:

* unsigned package: loads, reported `unsigned`;
* signed, no verifier registered: loads, reported `unverified` (never "verified");
* signed, verifier registered: must verify, otherwise refused;
* `STRATA_REQUIRE_SIGNED_COMPONENTS=1`: unsigned and unverifiable packages are refused (the updater turns it on while it checks a downloaded package).

## Building a package

    python packaging/package_engine.py --src <build folder> --backend sycl --version 0.1.45 \
        --out <components>/engine --arch intel-arc \
        [--build-id <id>] [--platform linux-x86_64] [--include SYSTEM-LIBS.txt | --tree] [--require-dsv41]
    python packaging/package_engine.py --verify <package folder>       # full hash check

Without `--include`/`--tree` it takes the entrypoint, the libraries beside it, `BUILD.json`, `LICENSE*`, `strata-vision` and
`strata-dsv41`. It refuses a package whose binaries import a library that is neither bundled nor the OS's or the GPU driver's
(`packaging/binary_deps.py`). The build id
defaults to the checkout's commit (`-dirty` when the tree has changes). It prints the size, file count and the SHA-256 of
the manifest file (which holds every file's hash), used as the package's digest in notes.

## What each target's package holds

| Package | Built on | Contents | Notes |
|---|---|---|---|
| `linux-x86_64` sycl, `archs` `intel-arc` | a Linux box with oneAPI DPC++ 2026.1 and `patchelf`, `packaging/build_sycl_linux.sh` (JIT from SPIR-V; ocloc cannot build the Alchemist FP64 kernels ahead of time, so one JIT package serves A-series and B-series) | `strata`, libsycl, the Unified Runtime adapters, the oneMKL SYCL libraries, libimf / libsvml / libirng / libintlc (about 650 MiB); every ELF file has RUNPATH `$ORIGIN` (Intel's libraries ship without it), so `env -i ./strata --version` runs | the Level Zero loader and compute runtime are the PC's driver; the user must be in the `render` group (`strata doctor` warns). No Windows package: `BACKEND_MATRIX` allows sycl on `linux-x86_64` only |

`packaging/collect_libs.py` gathers the Linux libraries (an `ldd` closure with the dlopen'd extras named) into the folder `package_engine.py` packages; libraries the distribution provides are listed in
`SYSTEM-LIBS.txt` inside the package so a missing one can be named on a target PC.

## The picture reader

At run time a reader that still cannot start tells why (stderr tail, exit status, and for a kernel-image error the card hint).
