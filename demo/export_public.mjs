import {writeFile} from "node:fs/promises";
import "./dist/study-data.js";
import "./dist/replay.js";

const source = globalThis.STOCKDESK;
const methods = Object.keys(source.cases[0].lead[1].models);
const seeds = [
  {id:"DEMO-STABLE",regime:"Stable",department:"Example", base:6},
  {id:"DEMO-VOLATILE",regime:"Volatile",department:"Example", base:5},
  {id:"DEMO-INTERMITTENT",regime:"Intermittent",department:"Example", base:2},
];
const days = source.history_dates.length + source.dates.length;
const pattern = [0,1,0,2,1,-2,-1];
const weights = {naive:1.25, seasonal_naive:1.2, moving_average:1.23, croston:1.12, sba:1.07};
const cases = seeds.map((seed,index) => {
  const series = Array.from({length:days},(_,day)=>{
    if (index===0) return seed.base + pattern[day%7] + (day%5===2 ? 2 : 0);
    if (index===1) return Math.max(0, seed.base + ((day*11+5)%17) - 8);
    return day%4===0 || day%9===3 ? seed.base + day%4 : 0;
  });
  const history = series.slice(0, source.history_dates.length);
  const item = {id:seed.id,regime:seed.regime,department:seed.department,history,
    demand:series.slice(history.length),lead:{}};
  for (const lead of [1,7]) {
    const horizon = lead + 1;
    const on_hand = index===0 ? 13 : horizon;
    const inbound = Array.from({length:lead},(_,i)=>i%2 ? 3 : 4);
    const models = {};
    item.lead[lead] = {on_hand,inbound,models};
    for (const model of methods) {
      const policy = source.cases[0].lead[lead].models[model];
      const median = [];
      const target = item.demand.map((_,day)=>{
        const seen = series.slice(0,history.length+day);
        const trail = seen.slice(-14);
        const average = trail.reduce((sum,value)=>sum+value,0)/trail.length;
        const positive = trail.filter(value=>value>0);
        const weekly = Array.from({length:horizon},(_,step)=>seen.at(-7+step%7)).reduce((sum,value)=>sum+value,0);
        const medianUnits = ({
          naive:seen.at(-1)*horizon, seasonal_naive:weekly, moving_average:average*horizon,
          croston:positive.length ? positive.reduce((sum,value)=>sum+value,0)/trail.length*horizon : 0,
          sba:average*horizon*.95,
        })[model] ?? (average*.65*horizon+weekly*.35);
        median.push(medianUnits);
        return Math.max(0, Math.round(medianUnits*(weights[model] ?? 1.19)+horizon*.45));
      });
      const {rule,alpha,feasible} = policy;
      models[model]={target,median,rule,alpha,feasible};
      const replay=new globalThis.StockReplay(item,lead,model);
      while (!replay.complete) replay.run(replay.review.suggested);
      models[model].golden=replay.totals;
    }
  }
  return item;
});

const {store,model,fill_floor,dates,history_dates,results,frontier,evidence,validation_winners} = source;
const published={store,model,fill_floor,dates,history_dates,results,frontier,evidence,validation_winners,cases,replayKind:"synthetic"};
await writeFile(new URL("./public/study-data.js",import.meta.url),"globalThis.STOCKDESK = "+JSON.stringify(published)+";\n");

