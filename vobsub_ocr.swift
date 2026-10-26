// Reads the text in DVD subtitle pictures with Apple's Vision framework, for
// vobsub.py, which compiles this on first use.
//
//     vobsub-ocr <language>    ISO 639-2, as in the track's tags ("eng"), or "und"
//
// stdin holds the pictures one after another: width and height as 32-bit
// big-endian integers, then width × height bytes of 8-bit gray, dark text on
// white. stdout gets one JSON string per picture, in order, with the lines of
// text joined by newlines, or "" where it found none.

import CoreGraphics
import Foundation
import Vision

let input = FileHandle.standardInput.readDataToEndOfFile()
var pictures: [(width: Int, height: Int, gray: Data)] = []
var at = input.startIndex
while at + 8 <= input.endIndex {
    let word = { (i: Int) in input[i..<i + 4].reduce(0) { $0 << 8 | Int($1) } }
    let (w, h) = (word(at), word(at + 4))
    pictures.append((w, h, input[at + 8..<at + 8 + w * h]))
    at += 8 + w * h
}

// Vision knows languages by BCP 47 tag ("en-US"); the track says "eng"
let alpha2 = Locale.LanguageCode(CommandLine.arguments.dropFirst().first ?? "und")
    .identifier(.alpha2)
let supported = (try? VNRecognizeTextRequest().supportedRecognitionLanguages()) ?? []
let language = alpha2.flatMap { code in
    supported.first { Locale.Language(identifier: $0).languageCode?.identifier == code }
}

/// Three times the size, with a margin: Vision reads DVD subtitles' 20-odd-pixel-
/// tall letters better enlarged. (At 2× it dropped a second line's leading "- ".)
func image(_ p: (width: Int, height: Int, gray: Data)) -> CGImage? {
    let gray = CGColorSpaceCreateDeviceGray()
    guard let provider = CGDataProvider(data: p.gray as CFData),
          let source = CGImage(width: p.width, height: p.height, bitsPerComponent: 8,
                               bitsPerPixel: 8, bytesPerRow: p.width, space: gray,
                               bitmapInfo: CGBitmapInfo(), provider: provider, decode: nil,
                               shouldInterpolate: true, intent: .defaultIntent)
    else { return nil }
    let (scale, margin) = (3, 32)
    let (w, h) = (p.width * scale + margin * 2, p.height * scale + margin * 2)
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: 0, space: gray, bitmapInfo: 0) else { return nil }
    ctx.setFillColor(gray: 1, alpha: 1)
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.interpolationQuality = .high
    ctx.draw(source, in: CGRect(x: margin, y: margin, width: p.width * scale,
                              height: p.height * scale))
    return ctx.makeImage()
}

func read(_ picture: CGImage) -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    if let language {
        request.recognitionLanguages = [language]
    } else {
        request.automaticallyDetectsLanguage = true
    }
    try? VNImageRequestHandler(cgImage: picture).perform([request])
    // Top to bottom (Vision's y runs upward), and pieces that share a line,
    // such as two speakers' "- Yes." "- No.", left to right
    let found = (request.results ?? []).compactMap { o in
        o.topCandidates(1).first.map { (box: o.boundingBox, text: $0.string) }
    }.sorted { $0.box.midY > $1.box.midY }
    var lines: [[(box: CGRect, text: String)]] = []
    for piece in found {
        if let last = lines.last?.last, abs(last.box.midY - piece.box.midY) < last.box.height / 2 {
            lines[lines.count - 1].append(piece)
        } else {
            lines.append([piece])
        }
    }
    return lines.map { $0.sorted { $0.box.minX < $1.box.minX }.map(\.text).joined(separator: " ") }
        .joined(separator: "\n")
}

var texts = [String](repeating: "", count: pictures.count)
texts.withUnsafeMutableBufferPointer { out in
    DispatchQueue.concurrentPerform(iterations: pictures.count) { i in
        if let picture = image(pictures[i]) { out[i] = read(picture) }
    }
}
for text in texts {
    let json = try! JSONSerialization.data(withJSONObject: text, options: .fragmentsAllowed)
    print(String(decoding: json, as: UTF8.self))
}
