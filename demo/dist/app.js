"use strict";

(() => {
  const data = globalThis.STOCKDESK;
  const get = id => document.getElementById(id);
  if (!data || typeof globalThis.StockReplay !== "function") {
    get("app-error").hidden = false;
    return;
  }
  const isSynthetic = data.replayKind === "synthetic";
  const demandKind = isSynthetic ? "synthetic demand" : "recorded demand";
  const number = new Intl.NumberFormat("en-US");
  const money = new Intl.NumberFormat("en-US", {style:"currency",currency:"USD",maximumFractionDigits:0});
  const pct = new Intl.NumberFormat("en-US", {style:"percent",minimumFractionDigits:1,maximumFractionDigits:2});
  const stock = new Intl.NumberFormat("en-US", {maximumFractionDigits:1});
  const dateFormat = new Intl.DateTimeFormat("en-US", {month:"short",day:"numeric",timeZone:"UTC"});
  const reviewDateFormat = new Intl.DateTimeFormat("en-US", {
    weekday:"long", month:"short", day:"numeric", timeZone:"UTC",
  });
  const fmtDate = value => dateFormat.format(new Date(value + "T00:00:00Z"));
  const addDays = (value, days) => {
    const date = new Date(value + "T00:00:00Z");
    date.setUTCDate(date.getUTCDate() + days);
    return date.toISOString().slice(0,10);
  };
  const modelNames = {naive:"Last observation", seasonal_naive:"Seasonal naïve", moving_average:"Moving average",
    croston:"Croston", sba:"SBA", [data.model]:"LightGBM"};
  const modelOrder = [data.model,"moving_average","naive","seasonal_naive","croston","sba"];
  const markers = {
    [data.model]: ["teal", "<circle r='6'/>"],
    moving_average: ["ink", "<rect x='-5' y='-5' width='10' height='10' rx='1'/>"],
    naive: ["blue", "<path d='M0 -7 7 0 0 7 -7 0Z'/>"],
    seasonal_naive: ["amber", "<path d='M0 -7 7 6 -7 6Z'/>"],
    croston: ["violet", "<path d='M-6 -3 0 -6 6 -3 6 3 0 6 -6 3Z'/>"],
    sba: ["rose", "<path d='M-2 -7H2V-2H7V2H2V7H-2V2H-7V-2H-2Z'/>"],
  };
  function marker(model, position = "", title = "") {
    const [color, shape] = markers[model];
    return "<g class='method-marker method-" + color + "'" + (position ? " transform='" + position + "'" : "") +
      ">" + (title ? "<title>" + title + "</title>" : "") + shape + "</g>";
  }
  const rules = {direct_quantile:"Predictive quantile",own_buffer:"Own safety buffer",shared_buffer:"Shared safety buffer"};
  const viewNames = {desk:"Order desk",results:"Study results",guide:"How it works"};
  const html = (target, value) => get(target).innerHTML = value;
  const set = (target, value) => get(target).textContent = value;
  const start = () => {
    if (isSynthetic) {
      get("public-replay-note").hidden = false;
      set("public-replay-note", "This hosted replay uses synthetic sales and illustrative targets. Study results show the verified CA_1 simulations.");
      set("history-caption", "Daily synthetic demand available before the current review.");
      set("history-label", "Synthetic units");
      set("guide-replay-scope", "The order desk uses three synthetic teaching items. The study results use recorded M5 sales.");
    }
    for (const item of data.cases) get("sku").add(new Option(item.id + " · " + item.regime, item.id));
    for (const name of modelOrder) get("model").add(new Option(modelNames[name], name));
  };

  let play, outcome, manual, view;
  function reset() {
    const item = data.cases.find(item => item.id === get("sku").value) || data.cases[0];
    play = new StockReplay(item, get("lead").value, get("model").value);
    outcome = null;
    manual = false;
    drawDesk();
  }

  const valueHTML = value => value + "<span>units</span>";
  function drawDesk() {
    const visibleDay = outcome ? outcome.day : Math.min(play.day, data.dates.length - 1);
    const reviewDate = data.dates[visibleDay];
    const table = outcome || play.review;
    get("review-date").dateTime = reviewDate;
    set("review-date", reviewDateFormat.format(new Date(reviewDate + "T00:00:00Z")));
    set("review-progress", (visibleDay + 1) + "/" + data.dates.length);
    set("item-regime", play.item.regime);
    html("available", valueHTML(number.format(table.available)));
    html("outstanding", valueHTML(number.format(table.outstanding)));
    html("target", valueHTML(stock.format(table.target)));
    html("suggestion", valueHTML(number.format(table.suggested)));
    set("target-caption", (play.lead + 1) + "-day protection period");
    set("arrival", "New units arrive " + fmtDate(addDays(reviewDate, play.lead)) + ", before demand.");
    get("decision-entry").hidden = !!outcome;
    get("day-result").hidden = !outcome;
    if (outcome) {
      set("decision-caption", play.complete ? "The April replay is complete." : isSynthetic ? "Synthetic demand for this day is revealed." : "Recorded sales for this day are revealed.");
      const lead = outcome.lost ? "Some " + demandKind + " went unfilled." : outcome.demand ? "All " + demandKind + " was filled." : "There was no " + demandKind + " that day.";
      html("day-outcome", "<strong>" + lead + "</strong><p>Demand: "+ number.format(outcome.demand) +
        " units · Ordered: " + number.format(outcome.quantity) + "</p><div class='outcome-counts'><div><b>" +
        number.format(outcome.filled) + "</b><small>Filled</small></div><div class='" + (outcome.lost ? "loss" : "") +
        "'><b>" + number.format(outcome.lost) + "</b><small>Lost</small></div></div>");
      set("next-day", play.complete ? "Start again" : "Next day");
    } else {
      set("decision-caption", "Choose how many units to order.");
      get("quantity").value = play.review.suggested;
      setMode(manual);
    }
    drawHistory();
    drawLedger();
  }

  function setMode(value) {
    manual = value;
    get("suggested-mode").setAttribute("aria-pressed", String(!manual));
    get("manual-mode").setAttribute("aria-pressed", String(manual));
    if (!manual) get("quantity").value = play.review.suggested;
    get("quantity").readOnly = !manual;
    get("minus").disabled = !manual;
    get("plus").disabled = !manual;
  }
  function nudge(delta) {
    get("quantity").value = Math.max(0, Math.min(1000000, (get("quantity").valueAsNumber || 0) + delta));
  }

  function drawLedger() {
    get("ledger-wrap").hidden = !play.log.length;
    const totals = play.totals;
    if (!play.log.length) {
      set("ledger-summary", "Completed reviews will appear here.");
      html("ledger-body", "");
      return;
    }
    set("ledger-summary", play.log.length + (play.log.length === 1 ? " review" : " reviews") + " · " +
      (totals.demand ? pct.format(totals.filled / totals.demand) + " unit fill" : "No " + demandKind) +
      " · " + number.format(totals.pipeline) + " units in transit");
    html("ledger-body", [...play.log].reverse().map(row =>
      "<tr><td>" + fmtDate(data.dates[row.day]) + "</td>" +
      [row.received,row.quantity,row.demand,row.filled,row.lost,row.stock].map(value => "<td>" + number.format(value) + "</td>").join("") + "</tr>").join(""));
  }

  function drawHistory() {
    const history = data.history_dates.map((date, i) => ({date, demand:play.item.history[i], latest:false}));
    const completed = play.log.map(row => ({date:data.dates[row.day],demand:row.demand,latest:outcome?.day === row.day}));
    const points = [...history, ...completed].slice(-14);
    const W=660, H=225, left=38, right=14, top=23, bottom=184;
    const max = Math.max(4, ...points.map(row => row.demand));
    const tick = Math.max(1,Math.ceil(max / 4));
    const limit = tick * 4;
    const step = (W-left-right) / (points.length+1);
    const y = amount => bottom - amount / limit * (bottom-top);
    let svg = "<svg viewBox='0 0 " + W + " " + H + "' role='img' aria-label='Daily " + demandKind + " from " + fmtDate(points[0].date) +
      " through " + fmtDate(points.at(-1).date) + "'><title>Daily " + demandKind + "</title>";
    for (let value=0; value<=limit; value+=tick) {
      svg += "<line x1='"+left+"' x2='"+(W-right)+"' y1='"+y(value)+"' y2='"+y(value)+"' stroke='var(--line)'/>"+
        "<text x='"+(left-8)+"' y='"+(y(value)+4)+"' text-anchor='end'>"+value+"</text>";
    }
    points.forEach((row,i) => {
      const x = left + (i+.5)*step;
      const height = Math.max(2,bottom-y(row.demand));
      svg += "<rect x='"+(x-step*.28)+"' y='"+(bottom-height)+"' width='"+(step*.56)+"' height='"+height+"' rx='2' fill='"+
        (row.latest ? "var(--accent)" : "var(--sage)")+"'><title>"+fmtDate(row.date)+": "+row.demand+" units</title></rect>";
      if ((i % 4 === 0 && i < points.length-2) || i === points.length-1) {
        svg += "<text x='"+x+"' y='205' text-anchor='middle'>"+fmtDate(row.date)+"</text>";
      }
    });
    if (!outcome) {
      const x = left + (points.length+.5)*step;
      svg += "<line x1='"+x+"' x2='"+x+"' y1='27' y2='"+bottom+"' stroke='var(--accent)' stroke-dasharray='4 4'/>"+
        "<text x='"+(W-right)+"' y='14' text-anchor='end' class='focus-label'>Today’s review</text>";
    }
    html("sales-chart",svg+"</svg>");
    set("history-range", fmtDate(points[0].date)+" – "+fmtDate(points.at(-1).date));
  }

  function drawResults() {
    const lead = Number(get("result-lead").value);
    const regime = get("regime").value;
    const phase = get("phase").value;
    const rows = data.results.filter(row => row.lead===lead && row.regime===regime && row.phase===phase);
    const front = data.frontier.filter(row => row.lead===lead && row.regime===regime && row.phase===phase);
    const ml = rows.find(row => row.model===data.model);
    const moving = rows.find(row => row.model==="moving_average");
    set("comparison-caption", (lead+1) + " days of protection · Base price-linked costs · Each dot is one selected policy");
    const visible = new Set(rows.map(row => row.model));
    html("chart-key", modelOrder.filter(model => visible.has(model)).map(model =>
      "<li><svg viewBox='-9 -9 18 18' aria-hidden='true'>" + marker(model) +
      "</svg>" + modelNames[model] + "</li>").join("") +
      (front.length ? "<li><span class='frontier-key' aria-hidden='true'></span>Cost–fill frontier</li>" : ""));
    const gap = (ml.fill_rate - moving.fill_rate)*100;
    const delta = ml.cost - moving.cost;
    const winner = modelNames[data.validation_winners[lead]];
    const context = (phase === "policy_validation" ? "Lowest-cost policy clearing the validation floor: " + winner + ". " :
      "Validation chose " + winner + " for this lead time. ");
    html("results-callout","<strong>"+context +"</strong> LightGBM is "+money.format(Math.abs(delta))+" "+
      (delta < 0 ? "lower" : "higher")+" in cost and "+Math.abs(gap).toFixed(2)+" percentage points "+
      (gap >= 0 ? "higher" : "lower")+" in unit fill than moving average in this view.");
    html("policy-body",[...rows].sort((a,b)=>a.cost-b.cost).map(row =>
      "<tr><td>"+modelNames[row.model]+"</td><td class='result-rule'>"+rules[row.rule]+"</td><td>"+pct.format(row.fill_rate)+
      "</td><td>"+number.format(row.lost)+"</td><td>"+stock.format(row.mean_on_hand)+"</td><td>"+money.format(row.cost)+
      "</td><td class='"+(row.validation_feasible ? "" : "not-met")+"'>"+(row.validation_feasible ? "Met floor" : "Below floor")+"</td></tr>").join(""));
    drawCostChart(rows,front);
  }

  function drawCostChart(rows, front) {
    const points = [...rows, ...front];
    const W=700,H=290,l=82,r=24,t=22,b=237;
    const floor=data.fill_floor;
    const minX=Math.max(0,Math.min(floor-.006, ...points.map(row=>row.fill_rate))-.007);
    const maxX=Math.min(1,Math.max(floor+.006,...points.map(row=>row.fill_rate))+.007);
    const tick=Math.max(50,Math.ceil(Math.max(...points.map(row=>row.cost))/4/50)*50);
    const topY=tick*4;
    const x=fill=>l+(fill-minX)/(maxX-minX)*(W-l-r);
    const y=cost=>b-cost/topY*(b-t);
    let svg="<svg viewBox='0 0 "+W+" "+H+"' role='img' aria-label='Cost and achieved unit fill by forecast. Values appear in the policy comparison table.'><title>Cost versus unit fill</title>";
    for(let i=0;i<=4;i++){
      const level=i*tick;
      const xx=l+(W-l-r)*i/4;
      const fraction=minX+(maxX-minX)*i/4;
      svg+="<line x1='"+l+"' x2='"+(W-r)+"' y1='"+y(level)+"' y2='"+y(level)+"' stroke='var(--line)'/><text x='"+(l-9)+
        "' y='"+(y(level)+4)+"' text-anchor='end'>"+money.format(level)+"</text><text x='"+xx+
        "' y='"+(b+23)+"' text-anchor='middle'>"+(fraction*100).toFixed(1)+"%</text>";
    }
    svg+="<line x1='"+x(floor)+"' x2='"+x(floor)+"' y1='"+t+"' y2='"+b+"' stroke='var(--accent)' stroke-dasharray='4 4'/><text x='"+
      (x(floor)+6)+"' y='15' class='focus-label'>"+(floor*100)+"% floor</text>";
    if(front.length) {
      const series=[...front].sort((a,b)=>a.fill_rate-b.fill_rate);
      const line=series.map(row=>x(row.fill_rate)+","+y(row.cost)).join(" ");
      svg+="<polyline points='"+line+"' fill='none' stroke='var(--sage)' stroke-width='2' stroke-dasharray='4 3'/>";
    }
    for (const row of rows) {
      svg += marker(row.model, "translate(" + x(row.fill_rate) + "," + y(row.cost) + ")",
        modelNames[row.model] + ": " + pct.format(row.fill_rate) + " fill, " + money.format(row.cost));
    }
    svg+="<text x='"+((W+l-r)/2)+"' y='281' class='axis-title' text-anchor='middle'>Achieved unit fill</text></svg>";
    html("comparison-chart",svg);
  }

  function route() {
    if (location.hash === "#main" && view) return;
    const candidate = location.hash.slice(1);
    const next = Object.hasOwn(viewNames,candidate) ? candidate : "desk";
    for (const name of Object.keys(viewNames)) get(name).hidden = name !== next;
    for (const anchor of document.querySelectorAll("[data-view]")) {
      if (anchor.dataset.view===next) anchor.setAttribute("aria-current","page");
      else anchor.removeAttribute("aria-current");
    }
    set("current-view",viewNames[next]);
    if (next==="results") drawResults();
    if (view && view!==next) window.scrollTo({top:0,behavior:"instant"});
    view=next;
  }

  start();
  reset();
  route();
  window.addEventListener("hashchange",route);
  for(const id of ["sku","lead","model"]) get(id).addEventListener("change",reset);
  for(const id of ["result-lead","regime","phase"]) get(id).addEventListener("change",drawResults);
  get("reset").addEventListener("click",reset);
  get("suggested-mode").addEventListener("click",()=>setMode(false));
  get("manual-mode").addEventListener("click",()=>{setMode(true);get("quantity").focus();get("quantity").select();});
  get("minus").addEventListener("click",()=>nudge(-1));
  get("plus").addEventListener("click",()=>nudge(1));
  get("order-form").addEventListener("submit",event=>{
    event.preventDefault();
    const quantity=manual ? get("quantity").valueAsNumber : play.review.suggested;
    if(!Number.isSafeInteger(quantity) || quantity<0 || quantity>1000000) return;
    outcome=play.run(quantity);
    drawDesk();
    get("next-day").focus();
  });
  get("next-day").addEventListener("click",()=>{
    if(play.complete) reset();
    else {outcome=null;drawDesk();}
    get("order-form").querySelector("button[type=submit]").focus();
  });
})();

