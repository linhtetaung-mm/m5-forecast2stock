import assert from "node:assert/strict";
import "./dist/study-data.js";
import "./dist/replay.js";

const data = globalThis.STOCKDESK;
assert.equal(data.dates.length,24);
assert.equal(data.cases.length,3);
let replayed=0;
for(const item of data.cases) {
  for(const lead of [1,7]) {
    for(const [model, expected] of Object.entries(item.lead[lead].models)) {
      const state = new globalThis.StockReplay(item, lead, model);
      while(!state.complete) state.run(state.review.suggested);
      assert.deepEqual(state.totals,expected.golden,item.id+" · "+lead+" · "+model);
      replayed++;
    }
  }
}
const item=data.cases[0];
const manager=new globalThis.StockReplay(item,7,data.model);
assert.throws(()=>manager.run(-1),/whole order/);
assert.throws(()=>manager.run(2.25),/whole order/);
const arrival=manager.review.available;
const placed=manager.run(44);
assert.equal(placed.available,arrival);
for(let i=1;i<7;i++) assert.notEqual(manager.run(0).received,44);
assert.equal(manager.run(0).received,44);
const result = data.results.find(row=>row.lead===7 && row.phase==="final_cutover" && row.regime==="All" && row.model===data.model);
assert.equal(result.fill_rate.toFixed(4),"0.9667");
assert.equal(Math.round(result.cost),4199);
console.log(replayed+" exported replay traces match the Python simulator.");

