import { describe, expect, test } from "bun:test";
import { handoffFile, trustedOrigin, urlFileName } from "./handoff";

describe("handoff", () => {
  test("only trusted origins", () => {
    expect(trustedOrigin("https://craftmatic.click")).toBe(true);
    expect(trustedOrigin("http://localhost:5173")).toBe(true);
    expect(trustedOrigin("http://127.0.0.1:8765")).toBe(true);
    expect(trustedOrigin("https://craftmatic.click.evil.com")).toBe(false);
    expect(trustedOrigin("https://example.com")).toBe(false);
  });
  test("load messages become model files", () => {
    const glb = new Uint8Array([103, 108, 84, 70]).buffer;
    const f = handoffFile({ type: "pokefit:load", name: "dir/build.glb", data: glb });
    expect(f?.name).toBe("build.glb");
    expect(f?.size).toBe(4);
    expect(handoffFile({ type: "pokefit:load", name: "x.exe", data: glb })).toBeNull();
    expect(handoffFile({ type: "pokefit:load", name: "x.glb", data: new ArrayBuffer(0) })).toBeNull();
    expect(handoffFile({ type: "other", name: "x.glb", data: glb })).toBeNull();
    expect(handoffFile("nope")).toBeNull();
  });
  test("url names", () => {
    expect(urlFileName("https://craftmatic.click/out/My%20House.glb?x=1")).toBe("My House.glb");
    expect(urlFileName("https://a.b/c.png")).toBeNull();
    expect(urlFileName("javascript:alert(1)//x.glb")).toBeNull();
  });
});
