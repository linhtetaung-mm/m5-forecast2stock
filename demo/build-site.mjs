import {copyFile, mkdir} from "node:fs/promises";
import "./public/study-data.js";
import "./dist/replay.js";
import assert from "node:assert/strict";

const data=globalThis.STOCKDESK;
assert.equal(data.replayKind,"synthetic");
assert.equal(data.cases.length,3);
assert.equal(data.results.length,144);
for (const item of data.cases) {
  assert.match(item.id,/^DEMO-/);
  for (const lead of [1,7]) {
    for (const [model, config] of Object.entries(item.lead[lead].models)) {
      const state=new globalThis.StockReplay(item,lead,model);
      while (!state.complete) state.run(state.review.suggested);
      assert.deepEqual(state.totals,config.golden);
    }
  }
}
const site=new URL("./site-build/",import.meta.url);
await mkdir(site,{recursive:true});
for (const file of ["index.html","app.js","replay.js","style3.css","icon.svg"]) {
  await copyFile(new URL("./dist/"+file,import.meta.url),new URL(file,site));
}
await copyFile(new URL("./public/study-data.js",import.meta.url),new URL("study-data.js",site));
console.log("Verified 36 synthetic replays and built Stockdesk in demo/site-build.");

