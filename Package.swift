// swift-tools-version: 6.0
// SPDX-License-Identifier: AGPL-3.0-only
// Ported from forge-sort-0.2.3 (AGPL-3.0).

import PackageDescription

let package = Package(
    name: "MetalRadixSort",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MetalRadixSort", targets: ["MetalRadixSort"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "MetalRadixSort",
            resources: [.process("Shaders")]
        ),
        .testTarget(
            name: "MetalRadixSortTests",
            dependencies: ["MetalRadixSort"]
        ),
    ]
)
