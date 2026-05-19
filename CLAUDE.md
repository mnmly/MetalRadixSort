# MetalRadixSort — agent instructions

## Documentation

`MetalRadixSort` ships DocC-generated reference docs (see
`Sources/MetalRadixSort/MetalRadixSort.docc/` and
`Scripts/build_docs.sh`). **`///` doc comments on public/`open` symbols
are published** to the static site that `Scripts/build_docs.sh` writes
into `docs/MetalRadixSort/`. If `REPO_URL` is set and the repo is pushed
to GitHub Pages, the docs are also browsable there.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if
  the *why* is non-obvious. Skip restating what the signature already
  says.
- Document each parameter with `- Parameter name:` (use the **internal**
  name when there's an external label — DocC warns otherwise).
- Cross-reference related symbols with double-backtick links, e.g.
  `` ``MetalRadixSortU64Pairs/encode(onto:keys:values:count:)`` ``.
  DocC link syntax is signature-sensitive: `foo(_:)` and `foo(_:_:)` are
  different.
- When you add a new top-level symbol that belongs in the curated
  sidebar, add it under the appropriate `## Topics` group in
  `Sources/MetalRadixSort/MetalRadixSort.docc/MetalRadixSort.md`. Topics
  are organized by *user task*, not alphabetic order.

Verify before declaring documentation work done:

```bash
Scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" or "external name used to
document parameter" warnings attributable to your changes.

## Build and test

`swift test` does **not** work — SwiftPM does not compile `.metal`
resources into a metallib. Use `xcodebuild`:

```bash
xcodebuild test \
  -scheme MetalRadixSort \
  -destination 'platform=macOS' \
  -derivedDataPath ./.xcdd
```

`STATUS.md` is the source of truth for current state, known limitations,
and queued future work (stability across equal keys, bit-range hint).
Read it before proposing changes to the sort kernel or wrapper.
