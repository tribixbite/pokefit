import { describe, expect, test } from "bun:test";
import { colourAt, guessName, levenshtein, mainColours, matchColour, type Pixels, samplePixels } from "./reference";

/** A picture like official art: a lit and a shaded tone, a small red eye with a lighter highlight pixel, a black
 * outline, anti-aliased pixels between them, and a transparent background. */
function art(): Pixels {
  const w = 120, h = 100;
  const data = new Uint8ClampedArray(4 * w * h);
  const put = (x: number, y: number, rgb: number[], a = 255): void => {
    data.set([...rgb, a], 4 * (y * w + x));
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x >= 100) continue;                                   // transparent strip
      if (x === 0 || y === 0 || x === 99 || y === 99) put(x, y, [1, 1, 1]);
      else if (y < 60) put(x, y, [0xF1, 0xE5, 0xA2]);           // lit cream
      else if (y === 60) put(x, y, [0xE6, 0xD8, 0x9A]);         // anti-aliased row between the tones
      else put(x, y, [0xDA, 0xCC, 0x93]);                       // shaded cream
    }
  }
  for (let y = 20; y < 25; y++) for (let x = 40; x < 45; x++) put(x, y, [0xC3, 0x69, 0x55]);   // eye, 25 px
  put(42, 22, [0xE4, 0x8C, 0x7D]);                              // its highlight
  put(110, 50, [0, 0, 255], 120);                               // half-transparent: ignored
  return { data, width: w, height: h };
}

describe("reference picture colours", () => {
  const px = art();
  const s = samplePixels(px);

  test("samples only opaque pixels", () => {
    expect(s.n).toBe(100 * 100);
  });

  test("main colours are exact colours of the picture, the small eye included", () => {
    const cols = mainColours(s).map((c) => c.hex);
    expect(cols[0]).toBe("#F1E5A2");
    expect(cols).toContain("#DACC93");
    expect(cols).toContain("#C36955");
    expect(cols).toContain("#010101");
    // No average of two tones (the anti-aliased row is 1 % of the picture: under its own cluster's mode or merged).
    expect(cols.filter((c) => c === "#E6D89A").length).toBeLessThanOrEqual(1);
  });

  test("a palette colour takes its family's most common exact colour", () => {
    expect(matchColour("#FFF9BF", s)?.hex).toBe("#F1E5A2");     // a file's cream -> the art's lit cream
    expect(matchColour("#C00D1E", s)?.hex).toBe("#C36955");     // a file's red -> the art's eye red
    expect(matchColour("#000000", s)?.hex).toBe("#010101");
  });

  test("a colour with nothing like it in the picture is not matched", () => {
    expect(matchColour("#0050FF", s)).toBeNull();
    expect(matchColour("#FFFFFF", samplePixels({ data: new Uint8ClampedArray(16), width: 2, height: 2 }))).toBeNull();
  });

  test("the colour at a pixel: its own tone, not the neighbourhood's average", () => {
    expect(colourAt(px, 42, 22, 2)).toBe("#E48C7D");             // the highlight pixel itself
    expect(colourAt(px, 41, 21, 2)).toBe("#C36955");
    expect(colourAt(px, 10, 30)).toBe("#F1E5A2");
    expect(colourAt(px, 115, 10)).toBeNull();                     // transparent
  });
});

describe("guessing the Pokémon from a file name", () => {
  const names = ["bulbasaur", "vulpix", "ninetales", "mr-mime", "mew", "mewtwo", "eevee", "type-null"];

  test("exact words and small typos", () => {
    expect(guessName("ninetales-50mb.stl Pokemon Ninetails", names)).toBe("ninetales");
    expect(guessName("Pokemon Ninetails", names)).toBe("ninetales");
    expect(guessName("vulpix_v2.3mf", names)).toBe("vulpix");
    expect(guessName("Mr. Mime figure", names)).toBe("mr-mime");
    expect(guessName("mewtwo.stl", names)).toBe("mewtwo");
  });

  test("short words must be exact; no guess without a match", () => {
    expect(guessName("mew.stl", names)).toBe("mew");
    expect(guessName("new.stl", names)).toBeNull();
    expect(guessName("dragon with egg", names)).toBeNull();
  });

  test("edit distance", () => {
    expect(levenshtein("ninetails", "ninetales")).toBe(2);
    expect(levenshtein("", "abc")).toBe(3);
    expect(levenshtein("eevee", "eevee")).toBe(0);
  });
});
