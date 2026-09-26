// 不啟動瀏覽器或 server，驗證前端 generation 中止與 AudioContext 排程。
const fs = require("fs");
const vm = require("vm");

async function main() {
  const html = fs.readFileSync("echo_stream/web/static/index.html", "utf8");
  const start = html.indexOf("let ctx = null");
  const end = html.indexOf("// ── Frame 解析", start);
  if (start < 0 || end < 0) throw new Error("找不到播放 helper 區段");

  const pendingDecodes = [];
  const startedSources = [];
  const sandbox = {
    pendingDecodes,
    startedSources,
    fetch: async () => ({ ok: true }),
    setInterval: () => 1,
    clearInterval: () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(html.slice(start, end), sandbox);
  vm.runInContext(`ctx = {
    currentTime: 10,
    destination: {},
    decodeAudioData: () => new Promise(resolve =>
      pendingDecodes.push(() => resolve({ duration: 5 }))),
    createBufferSource: () => ({
      connect() {},
      start(at) { this.at = at; startedSources.push(this); },
      stop() { this.onended?.(); },
    }),
    createGain: () => ({
      connect() {},
      gain: { setValueAtTime() {}, linearRampToValueAtTime() {} },
    }),
  };`, sandbox);

  vm.runInContext(
    "beginPlayback({generation:'old', turn_id:'turn-1'});" +
      "playChunk('old', 0, 5, new Uint8Array([1]));",
    sandbox,
  );
  for (let i = 0; i < 20 && pendingDecodes.length === 0; i++) {
    await new Promise(setImmediate);
  }
  if (pendingDecodes.length === 0) throw new Error("舊 generation 沒有開始解碼");

  vm.runInContext("abortPlayback('old');", sandbox);
  pendingDecodes.shift()();
  await vm.runInContext("playChain", sandbox);
  if (startedSources.length !== 0) throw new Error("中止後的舊解碼仍排入播放");

  vm.runInContext(
    "ctx.decodeAudioData = async () => ({duration: 0.5});" +
      "beginPlayback({generation:'new', turn_id:'turn-2'});" +
      "playChunk('new', 0, 0.5, new Uint8Array([1]));",
    sandbox,
  );
  await vm.runInContext("playChain", sandbox);
  if (startedSources.length !== 1 || startedSources[0].at > 10.02) {
    throw new Error("新 generation 沿用舊排程尾端");
  }

  console.log("fake AudioContext: 舊解碼已中止，新 generation 立即排程");
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
