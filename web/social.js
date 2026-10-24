/* Jake's people — ego-orbit side panel + brushing into the main graph.
   Loaded as a separate script after index.html's inline script; shares its
   globals: $, api, cy, lens, curTheme, curDays, inspect, showInsight, zoomCausal. */
const SOCIAL_COLORS = {Jake: "#f0883e", Alice: "#58a6ff", Tasha: "#3fb950",
  Lucia: "#bc8cff", Katrina: "#e3b341", Shure: "#79c0ff"};
const WEARER_IDX = {};
["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"].forEach((n, i) => WEARER_IDX[n] = i);

let pcy = null, socialData = null, selPerson = null, selData = null;
let socialTheme = null, socialDays = "1-7";

const PCY_STYLE = [
  {selector: "node", style: {
    "background-color": n => SOCIAL_COLORS[n.data("wearer")] || "#8b949e",
    "width": n => n.data("size"), "height": n => n.data("size"),
    "label": "data(label)", "color": "#e6edf3", "font-size": 10,
    "text-valign": "bottom", "text-margin-y": 4, "text-wrap": "wrap",
    "text-max-width": "110px"}},
  {selector: "node[?subject]", style: {"border-width": 4, "border-color": "#ffd33d"}},
  {selector: "node.sel", style: {"border-width": 4, "border-color": "#e6edf3"}},
  {selector: "edge", style: {"curve-style": "bezier"}},
  {selector: "edge.jake", style: {"line-color": "#8b949e",
    "width": e => e.data("w") || 1}},
  {selector: "edge.other", style: {"line-color": "#6e7681", "opacity": 0.25,
    "width": e => e.data("w") || 1}},
];

function socialInit() {
  const box = $("#people");
  if (!box) return;
  box.innerHTML = `<div id="people-hdr"><b>Jake's people</b><span id="people-filter"></span></div>
    <div id="pcy"></div>
    <div id="people-legend">closer = stronger bond · size = share of talk</div>`;
  pcy = cytoscape({container: $("#pcy"), elements: [], style: PCY_STYLE,
    wheelSensitivity: 0.3, minZoom: 0.3, maxZoom: 3,
    boxSelectionEnabled: false});
  pcy.on('tap', 'node', ev => selectPerson(ev.target.data('wearer')));
  pcy.on('tap', 'edge', ev => showPair(ev.target.data('a'), ev.target.data('b')));
  pcy.on('tap', ev => { if (ev.target === pcy) clearPerson(); });
  socialRefresh();
}

async function socialRefresh() {
  let url = `/api/social?days=${socialDays}`;
  if (socialTheme) url += `&theme=${socialTheme}`;
  socialData = await api(url);
  renderPeople();
}

function renderHeader() {
  const t = socialData && socialData.theme;
  $("#people-filter").innerHTML = t
    ? `· in <b>${t.name}</b><span id="pf-clear" title="clear theme filter">✕</span>` : "";
  const x = $("#pf-clear");
  if (x) x.onclick = () => socialFocusTheme(null);
}

function renderPeople() {
  if (!pcy || !socialData) return;
  renderHeader();
  const d = socialData, theme = d.theme;
  const metric = p => theme ? (p.participation || 0) : (p.share || 0);
  const mx = Math.max(...d.people.map(metric), 1e-6);
  const pairMap = {};
  for (const q of d.pairs || []) pairMap[q.a + "|" + q.b] = q;
  const get = (a, b) => WEARER_IDX[a] < WEARER_IDX[b]
    ? pairMap[a + "|" + b] : pairMap[b + "|" + a];
  const maxCo = Math.max(...(d.pairs || []).map(q => q.co_events || 0), 1);
  const els = [];
  const jake = d.people.find(p => p.name === "Jake");
  if (jake)
    els.push({data: {id: "p-Jake", wearer: "Jake", subject: 1,
                     label: "Jake\n" + (jake.badge ? jake.badge.text : ""),
                     size: 24 + 40 * Math.sqrt(metric(jake) / mx)},
              position: {x: 0, y: 0}});
  d.people.filter(p => p.name !== "Jake").forEach((p, i, others) => {
    const q = get("Jake", p.name);
    const bondN = theme ? 100 * ((q && q.co_events) || 0) / maxCo : (q ? q.bond : 0);
    const ang = -Math.PI / 2 + i * 2 * Math.PI / others.length;
    const rad = 70 + (1 - bondN / 100) * 130;
    els.push({data: {id: "p-" + p.name, wearer: p.name,
                     label: p.name + "\n" + (p.badge ? p.badge.text : ""),
                     size: 24 + 40 * Math.sqrt(metric(p) / mx)},
              position: {x: rad * Math.cos(ang), y: rad * Math.sin(ang)}});
    els.push({data: {id: "pe-Jake-" + p.name, source: "p-Jake",
                     target: "p-" + p.name, a: "Jake", b: p.name,
                     w: theme ? 1 + 8 * bondN / 100 : 1 + bondN / 12},
              classes: "jake"});
  });
  for (const q of d.pairs || []) {
    if (q.a === "Jake" || q.b === "Jake") continue;
    if ((q.bond || 0) < 40) continue;
    els.push({data: {id: `pe-${q.a}-${q.b}`, source: "p-" + q.a, target: "p-" + q.b,
                     a: q.a, b: q.b, w: 1 + q.bond / 25},
              classes: "other"});
  }
  pcy.elements().remove();
  pcy.add(els);
  pcy.fit(undefined, 30);
  if (selPerson) pcy.getElementById("p-" + selPerson).addClass("sel");
}

// ---------- brushing ----------

function ensureSdim() {
  if (cy && !cy._sdim) {
    cy.style().selector(".sdim").style({"opacity": 0.12}).update();
    cy._sdim = true;
  }
}

function applyPersonWeight(d) {
  if (!cy || !d) return;
  const ts = d.theme_share || [];
  const mx = Math.max(...ts.map(t => t.share || 0), 1e-6);
  const byId = {};
  ts.forEach(t => byId[t.theme_id] = t.share || 0);
  cy.nodes().forEach(n => {
    if (n.data("type") === "THEME") {
      const tid = +String(n.data("id")).replace(/^t/, "");
      const r = (byId[tid] || 0) / mx;
      n.style({opacity: 0.2 + 0.8 * r, "border-width": 2 + 6 * r});
    }
  });
  const evset = new Set((d.event_ids || []).map(id => "e" + id));
  ensureSdim();
  cy.nodes().forEach(n => {
    if (n.data("type") === "EVENT")
      n.toggleClass("sdim", evset.size > 0 && !evset.has(n.data("id")));
  });
}

function clearPersonWeight() {
  if (!cy) return;
  cy.nodes().forEach(n => {
    if (n.data("type") === "THEME") n.style({opacity: 1, "border-width": 2});
    if (n.data("type") === "EVENT") n.removeClass("sdim");
  });
}

async function selectPerson(name) {
  selPerson = name;
  if (pcy) {
    pcy.nodes().removeClass("sel");
    pcy.getElementById("p-" + name).addClass("sel");
  }
  selData = await api(`/api/social/person/${name}`);
  renderPersonCard(selData);
  applyPersonWeight(selData);
}

function clearPerson() {
  selPerson = null;
  selData = null;
  if (pcy) pcy.nodes().removeClass("sel");
  clearPersonWeight();
}

// ---------- inspector cards ----------

const effRow = e => `<div class="mention">${e.text}
  <span style="color:var(--dim)"> · z=${(+e.z).toFixed(1)}
  ${e.p_with != null ? ` · ${Math.round(100 * e.p_with)}% vs ${Math.round(100 * e.p_without)}%` : ""}</span></div>`;

function renderPersonCard(d) {
  const insp = $("#inspector");
  if (d.error) { insp.innerHTML = `<p>${d.error}</p>`; return; }
  const perday = (d.per_day || []).map(x => `D${x.day} ${(x.lines / 1000).toFixed(1)}k`).join(" · ");
  const pats = (d.patterns || []).map(p =>
    `<div class="mention"><span style="color:#39c5cf">${p.direction === "out" ? d.name + " → " + p.other : p.other + " → " + d.name}</span>
     <div class="rat">${p.rationale || ""}</div></div>`).join("");
  const tshare = (d.theme_share || []).filter(t => t.share > 0)
    .sort((a, b) => b.share - a.share).map(t =>
      `<div class="mention">${t.name} <span style="color:var(--dim)">${t.n_with}/${t.n_events}</span></div>`).join("");
  insp.innerHTML = `<h2 style="color:${SOCIAL_COLORS[d.name] || "#e6edf3"}">◉ ${d.name}</h2>
    <div class="meta">${d.code} · ${d.lines.toLocaleString()} lines · ${(100 * d.share).toFixed(1)}% of talk
      · rank #${d.rank} of 6 · ${Math.round(d.airtime_s / 60)} min airtime</div>
    <div class="meta">${perday}</div>
    <h3>bonds</h3>${(d.bonds || []).map(b =>
      `<div class="bondbar"><span class="bn">${b.other}</span>
        <span class="btr"><span class="bf" style="width:${b.bond}%"></span></span>
        <span class="bv">${b.bond}</span></div>`).join("") || "—"}
    <h3>how ${d.name} acts around Jake</h3>${(d.effects_around_jake || []).map(effRow).join("") || "—"}
    <h3>how Jake acts around ${d.name}</h3>${(d.jake_around_them || []).map(effRow).join("") || "—"}
    <h3>self effects (vs group)</h3>${(d.self_effects || []).map(effRow).join("") || "—"}
    <h3>interaction patterns</h3>${pats || "—"}
    <h3>related insights</h3>${(d.insights || []).map(r =>
      `<div class="edgecard relins" data-i="${r.id}">${r.title}
        <span style="color:var(--dim)"> ${r.lens}</span></div>`).join("") || "—"}
    <h3>theme share</h3>${tshare || "—"}
    <div class="meta" style="margin-top:10px">tap the people-panel background to clear</div>`;
  insp.querySelectorAll(".relins").forEach(c =>
    c.onclick = () => showInsight(+c.dataset.i));
}

async function showPair(a, b) {
  const d = await api(`/api/social/pair?a=${a}&b=${b}`);
  const insp = $("#inspector");
  if (d.error) { insp.innerHTML = `<p>${d.error}</p>`; return; }
  const t = d.totals || {};
  insp.innerHTML = `<h2>${d.a} ↔ ${d.b}</h2>
    <div class="meta">bond ${d.bond}/100 · ${t.copresence_min || 0} min together · ${t.replies || 0} replies</div>
    <h3>per day — copresence · replies · talk-overs</h3>
    <table class="ptable"><tr><th>day</th><th>cop</th>
      <th>${d.a}→${d.b}</th><th>${d.b}→${d.a}</th>
      <th>${d.a}▸${d.b}</th><th>${d.b}▸${d.a}</th></tr>
      ${(d.per_day || []).map(r =>
        `<tr><td>D${r.day}</td><td>${r.copresence_min}</td>
         <td>${r.a_replies_b}</td><td>${r.b_replies_a}</td>
         <td>${r.a_over_b}</td><td>${r.b_over_a}</td></tr>`).join("")}
    </table>
    ${(d.effects || []).length ? `<h3>effects</h3>` + d.effects.map(e =>
      `<div class="mention"><b>${e.subject}</b> ${e.text}
        <span style="color:var(--dim)"> around ${e.other} · z=${(+e.z).toFixed(1)}</span></div>`).join("") : ""}
    ${(d.patterns || []).length ? `<h3>patterns</h3>` + d.patterns.map(p =>
      `<div class="mention"><span style="color:#39c5cf">${p.direction}</span>
        <span class="rat"> ${p.rationale || ""}</span></div>`).join("") : ""}
    ${(d.shared_events || []).length ? `<h3>shared events</h3>` + d.shared_events.map(e =>
      `<div class="edgecard" data-node="e${e.id}"><b>D${e.day} ${e.hms}</b> ${e.summary}</div>`).join("") : ""}
    ${(d.insights || []).length ? `<h3>related insights</h3>` + d.insights.map(r =>
      `<div class="edgecard relins" data-i="${r.id}">${r.title}
        <span style="color:var(--dim)"> ${r.lens}</span></div>`).join("") : ""}`;
  insp.querySelectorAll(".edgecard[data-node]").forEach(c => c.onclick = () => {
    const nid = c.dataset.node, n = cy && cy.getElementById(nid);
    if (n && n.nonempty())
      cy.animate({center: {eles: n}, zoom: cy.zoom()}, {duration: 300});
    inspect(nid.slice(1));
  });
  insp.querySelectorAll(".relins").forEach(c =>
    c.onclick = () => showInsight(+c.dataset.i));
}

// ---------- hooks called from the inline script ----------

async function socialFocusTheme(tid) {
  tid = tid || null;
  const cur = socialData && socialData.theme ? socialData.theme.id : null;
  socialTheme = tid;
  if (tid === cur) { renderHeader(); return; }
  await socialRefresh();
}

async function socialFocusDays(d) {
  socialDays = d || "1-7";
  await socialRefresh();
}

function socialAfterRender() {
  if (selPerson && selData) applyPersonWeight(selData);
}
