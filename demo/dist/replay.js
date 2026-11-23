"use strict";

globalThis.StockReplay = class StockReplay {
  constructor(item, lead, model) {
    this.item = item;
    this.lead = Number(lead);
    const stock = item.lead[this.lead];
    this.model = stock.models[model];
    this.onHand = stock.on_hand;
    this.inbound = [...stock.inbound];
    this.day = 0;
    this.log = [];
  }

  get complete() {
    return this.day === this.item.demand.length;
  }

  get review() {
    if (this.complete) return null;
    const received = this.inbound[0];
    const available = this.onHand + received;
    const outstanding = this.inbound.slice(1).reduce((sum, quantity) => sum + quantity, 0);
    const target = this.model.target[this.day];
    return {received, available, outstanding, position: available + outstanding, target,
      suggested: Math.max(0, Math.ceil(target - available - outstanding))};
  }

  run(quantity) {
    if (this.complete) throw new Error("The replay is complete.");
    if (!Number.isSafeInteger(quantity) || quantity < 0 || quantity > 1000000) {
      throw new Error("Use a whole order quantity from 0 to 1,000,000.");
    }
    const today = this.review;
    const demand = this.item.demand[this.day];
    const filled = Math.min(demand, today.available);
    this.onHand = today.available - filled;
    this.inbound.shift();
    this.inbound.push(quantity);
    const result = {...today, day: this.day, quantity, demand, filled, lost: demand - filled, stock: this.onHand};
    this.log.push(result);
    this.day += 1;
    return result;
  }

  get totals() {
    const totals = this.log.reduce((sum, row) => ({
      demand: sum.demand + row.demand, filled: sum.filled + row.filled,
      lost: sum.lost + row.lost, ordered: sum.ordered + row.quantity,
    }), {demand:0, filled:0, lost:0, ordered:0});
    return {...totals, stock: this.onHand, pipeline: this.inbound.reduce((sum, quantity) => sum + quantity, 0)};
  }
};

