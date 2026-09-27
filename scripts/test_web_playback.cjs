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

  await testBackendStopReportsFinalPosition();
  await testAckRetry();
}

// 建立獨立 sandbox：可控的 fetch 回應與計時器，不依賴真實網路。
function makeSandbox(html, fetchImpl) {
  const start = html.indexOf("let ctx = null");
  const end = html.indexOf("// ── Frame 解析", start);
  const timers = [];
  const calls = [];
  const sandbox = {
    calls,
    timers,
    fetch: (url, init) => {
      calls.push({ url, body: JSON.parse(init.body) });
      return fetchImpl(calls.length);
    },
    setInterval: () => 1,
    clearInterval: () => {},
    setTimeout: (fn) => { timers.push(fn); return timers.length; },
    clearTimeout: () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(html.slice(start, end), sandbox);
  vm.runInContext(`ctx = {
    currentTime: 10,
    destination: {},
    decodeAudioData: async () => ({ duration: 2 }),
    createBufferSource: () => ({
      connect() {},
      start(at) { this.at = at; },
      stop() { this.onended?.(); },
    }),
    createGain: () => ({
      connect() {},
      gain: { setValueAtTime() {}, linearRampToValueAtTime() {} },
    }),
  };`, sandbox);
  return sandbox;
}

async function flush() {
  for (let i = 0; i < 20; i++) await new Promise(setImmediate);
}

async function testBackendStopReportsFinalPosition() {
  const html = fs.readFileSync("echo_stream/web/static/index.html", "utf8");
  const sb = makeSandbox(html, async () => ({ ok: true, status: 200 }));
  vm.runInContext(
    "beginPlayback({generation:'g1', turn_id:'t1'});" +
      "playChunk('g1', 0, 2, new Uint8Array([1]));",
    sb,
  );
  await vm.runInContext("playChain", sb);
  // 播到 11.02 秒時後端發出停止（起播點 10.02）→ 實播約 1 秒
  vm.runInContext("ctx.currentTime = 11.02; abortPlayback({generation:'g1', turn_id:'t1'});", sb);
  await flush();
  const final = sb.calls.find(c => c.body.status === "aborted");
  if (!final) throw new Error("後端發起停止時前端沒有回報最終播放位置");
  if (final.body.generation !== "g1" || Math.abs(final.body.played_s - 1) > 0.01) {
    throw new Error(`最終播放位置錯誤: ${JSON.stringify(final.body)}`);
  }
  // 已停止的 generation 若有遲到的 BEGIN/AUDIO/END，不得復活播放狀態
  vm.runInContext(
    "beginPlayback({generation:'g1', turn_id:'t1'});" +
      "endPlayback({generation:'g1', turn_id:'t1', chunks:1, duration_s:2});",
    sb,
  );
  if (vm.runInContext("playbacks.has('g1')", sb)) throw new Error("已停止的 generation 被遲到 frame 復活");
  console.log("後端停止：前端回報最終播放位置，遲到 frame 被忽略");
}

async function testAckRetry() {
  const html = fs.readFileSync("echo_stream/web/static/index.html", "utf8");
  // 第一次網路錯誤、第二次 5xx、第三次成功
  const sb = makeSandbox(html, async (n) => {
    if (n === 1) throw new Error("network down");
    if (n === 2) return { ok: false, status: 502 };
    return { ok: true, status: 200 };
  });
  vm.runInContext(
    "beginPlayback({generation:'g2', turn_id:'t2'});" +
      "endPlayback({generation:'g2', turn_id:'t2', chunks:0, duration_s:0});",
    sb,
  );
  await flush();
  for (let i = 0; i < 5 && sb.calls.length < 3; i++) {
    const pending = sb.timers.splice(0);
    pending.forEach(fn => fn());
    await flush();
  }
  const acks = sb.calls.filter(c => c.body.status === "complete");
  if (acks.length !== 3) throw new Error(`完成 ACK 失敗後未重試到成功: ${acks.length} 次`);
  if (acks.some(c => c.body.generation !== "g2")) throw new Error("重試的 ACK generation 不一致");
  sb.timers.splice(0).forEach(fn => fn());
  await flush();
  if (sb.calls.length !== 3) throw new Error("ACK 成功後仍持續重送");

  // 後端明確拒絕（409）不重試
  const rejected = makeSandbox(html, async () => ({ ok: false, status: 409 }));
  vm.runInContext(
    "beginPlayback({generation:'g3', turn_id:'t3'});" +
      "endPlayback({generation:'g3', turn_id:'t3', chunks:0, duration_s:0});",
    rejected,
  );
  await flush();
  rejected.timers.splice(0).forEach(fn => fn());
  await flush();
  if (rejected.calls.length !== 1) throw new Error("後端 409 拒絕後仍重試");
  console.log("完成 ACK：失敗有限次重試、成功才清理、409 不重試");
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
