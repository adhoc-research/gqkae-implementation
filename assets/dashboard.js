/* GQKAE figures - builds light-theme ECharts panels from window.GQKAE_DATA */
(function () {
  "use strict";

  var D = window.GQKAE_DATA;
  if (!D) {
    console.error("GQKAE_DATA missing - run scripts/build_site_data.py");
    return;
  }

  // ---- light academic palette -------------------------------------------
  var C = {
    blue: "#2b6cb0", green: "#2f855a", orange: "#c05621", violet: "#6b46c1",
    red: "#c53030",
    ink: "#1a1a1a", muted: "#6a6f76", faint: "#9aa0a6",
    grid: "#e7e7e3", tip: "#ffffff", tipBorder: "#d9dad5",
    rampBlue: ["#eef3f8", "#2b6cb0"], rampViolet: ["#f0ecf8", "#6b46c1"],
  };
  var charts = [];

  function tip(extra) {
    return Object.assign({
      backgroundColor: C.tip, borderColor: C.tipBorder, borderWidth: 1,
      textStyle: { color: C.ink, fontSize: 12 },
      extraCssText: "box-shadow:0 2px 10px rgba(0,0,0,0.08);",
    }, extra || {});
  }
  // generous default margins; legend sits in a reserved band at the top
  function base(opt) {
    return Object.assign({
      backgroundColor: "transparent",
      textStyle: { color: C.ink, fontFamily: "Inter, sans-serif" },
      grid: { left: 22, right: 30, top: 58, bottom: 22, containLabel: true },
      tooltip: tip(),
      legend: {
        type: "scroll", top: 10, left: "center",
        textStyle: { color: C.muted, fontSize: 11 },
        icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18,
      },
    }, opt);
  }
  function mkAxis(name, loc, gap, extra) {
    return Object.assign({
      name: name, nameLocation: loc, nameGap: gap,
      nameTextStyle: { color: C.faint, fontSize: 11 },
      axisLine: { lineStyle: { color: C.grid } },
      axisLabel: { color: C.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: C.grid, type: "dashed" } },
      axisTick: { show: false },
    }, extra || {});
  }
  function xaxis(name, extra) { return mkAxis(name, "middle", name ? 32 : 0, extra); }
  function yaxis(name, extra) { return mkAxis(name, "middle", name ? 52 : 0, extra); }

  function make(id, option) {
    var el = document.getElementById(id);
    if (!el) return;
    var ch = echarts.init(el, null, { renderer: "canvas" });
    ch.setOption(option);
    charts.push(ch);
    return ch;
  }
  function fmt(x, d) { return x == null ? "n/a" : Number(x).toFixed(d == null ? 3 : d); }
  function sci(x) { return x == null ? "n/a" : Number(x).toExponential(2); }
  function abbr(x) {
    if (x == null) return "n/a";
    var n = Number(x);
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(n);
  }
  // label only alternating decades of a log axis (thins clutter)
  function decadeLabel(v) {
    var e = Math.round(Math.log10(v));
    return e % 2 === 0 ? "10" + supExp(e) : "";
  }
  function supExp(e) {
    var map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    return String(e).split("").map(function (c) { return map[c] || c; }).join("");
  }

  var pes = D.pes, bonds = pes.bonds, pts = pes.points;
  var Rs = bonds.map(function (b) { return b.R; });
  var Rlabels = Rs.map(function (r) { return r.toFixed(2); });
  var T = D.paper_targets;

  // ---- summary key-figures + setup --------------------------------------
  function text(id, v) { var e = document.getElementById(id); if (e) e.textContent = v; }
  var ov = pes.overall;
  text("stat-jobs", ov.n_pass + " / " + ov.n_jobs);
  text("stat-err", sci(ov.mean_abs_error));
  text("stat-cx", fmt(ov.mean_two_qubit, 1));
  text("stat-runtime", (ov.total_runtime_s / 3600).toFixed(2));

  var env = D.meta.environment;
  text("pv-gpu", env.gpu || "n/a");
  text("pv-cudaq", (env.cudaq_version || "").replace("CUDA-Q Version ", "").split(" ")[0] || "n/a");
  text("pv-shots", ov.shots ? ov.shots.toLocaleString() : "n/a");
  text("pv-iters-batch", (ov.iterations || "n/a") + " × " + (ov.batch_circuits || "n/a"));
  text("pv-grid", Rs.length + " bond lengths × " + pes.seeds.length + " seeds");
  text("pv-env", env.python ? "Python " + env.python : "");
  text("meta-commit", D.meta.commit ? "@" + D.meta.commit : "");
  text("meta-generated", D.meta.generated ? D.meta.generated.replace("T", " ").replace("+00:00", " UTC") : "");

  // ---- PES curve --------------------------------------------------------
  make("chart-pes", base({
    grid: { left: 16, right: 28, top: 56, bottom: 16, containLabel: true },
    legend: { type: "scroll", top: 10, left: "center", textStyle: { color: C.muted, fontSize: 11 },
      icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18,
      data: ["CASCI", "GQKAE mean", "Hartree-Fock", "per-seed"] },
    tooltip: tip({ trigger: "axis", valueFormatter: function (v) { return v == null ? "n/a" : v.toFixed(6) + " Ha"; } }),
    xAxis: xaxis("bond length R (Å)", { type: "category", data: Rlabels, boundaryGap: false }),
    yAxis: yaxis("energy (Ha)", { type: "value", scale: true }),
    series: [
      { name: "CASCI", type: "line", smooth: true, symbol: "none", lineStyle: { width: 3, color: C.green },
        data: bonds.map(function (b) { return b.casci; }) },
      { name: "GQKAE mean", type: "line", smooth: true, symbol: "circle", symbolSize: 7,
        lineStyle: { width: 2, type: "dashed", color: C.blue }, itemStyle: { color: C.blue },
        data: bonds.map(function (b) { return b.gqkae.mean; }) },
      { name: "Hartree-Fock", type: "line", smooth: true, symbol: "none", lineStyle: { width: 1.5, color: C.faint, type: "dotted" },
        data: bonds.map(function (b) { return b.hf; }) },
      { name: "per-seed", type: "scatter", symbolSize: 5, itemStyle: { color: "rgba(43,108,176,0.35)" },
        data: pts.map(function (p) { return [p.R.toFixed(2), p.gqkae]; }) },
    ],
  }));

  // ---- accuracy: |error| vs R (log) -------------------------------------
  var chem = ov.chem_accuracy_ha;
  make("chart-error", base({
    legend: { type: "scroll", top: 10, left: "center", textStyle: { color: C.muted, fontSize: 11 },
      icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18, data: ["mean |error|", "per-seed |error|"] },
    tooltip: tip({ trigger: "item", formatter: function (p) { return "R=" + p.value[0] + " Å<br/>|err| = " + Number(p.value[1]).toExponential(2) + " Ha"; } }),
    xAxis: xaxis("bond length R (Å)", { type: "value", min: 0.4, max: 1.4 }),
    yAxis: yaxis("|error| vs CASCI (Ha)", { type: "log", min: 1e-16, max: 1e-2,
      minorTick: { show: false }, minorSplitLine: { show: false },
      axisLabel: { color: C.muted, fontSize: 11, formatter: decadeLabel } }),
    series: [
      { name: "chem. accuracy", type: "line", data: [], markLine: {
          silent: true, symbol: "none", lineStyle: { color: C.orange, type: "dashed", width: 1.5 },
          label: { color: C.orange, formatter: "chemical accuracy 1.6 mHa", position: "insideEndTop", fontSize: 10 },
          data: [{ yAxis: chem }] } },
      { name: "per-seed |error|", type: "scatter", symbolSize: 7, itemStyle: { color: "rgba(107,70,193,0.5)" },
        data: pts.map(function (p) { return [p.R, Math.max(p.abs_error, 1e-16)]; }) },
      { name: "mean |error|", type: "line", smooth: true, symbol: "diamond", symbolSize: 9,
        lineStyle: { color: C.blue, width: 2 }, itemStyle: { color: C.blue },
        data: bonds.map(function (b) { return [b.R, Math.max(b.abs_error.mean, 1e-16)]; }) },
    ],
  }));

  // ---- heatmaps ---------------------------------------------------------
  heatmap("chart-err-heat", "abs_error", "|error| (Ha)", true, C.rampViolet);
  heatmap("chart-cx-heat", "two_qubit", "CX gates", false, C.rampBlue);

  function heatmap(id, field, title, isLog, colors) {
    var seeds = pes.seeds;
    var data = [], vals = [];
    pts.forEach(function (p) {
      var ri = Rs.indexOf(p.R), si = seeds.indexOf(p.seed);
      var v = p[field];
      data.push([ri, si, v]);
      vals.push(isLog ? Math.log10(Math.max(v, 1e-16)) : v);
    });
    var vmin = Math.min.apply(null, vals), vmax = Math.max.apply(null, vals);
    make(id, base({
      grid: { left: 12, right: 16, top: 18, bottom: 64, containLabel: true },
      tooltip: tip({ position: "top", formatter: function (p) {
          var v = pts.filter(function (q) { return Rs.indexOf(q.R) === p.value[0] && seeds.indexOf(q.seed) === p.value[1]; })[0];
          var raw = v ? v[field] : null;
          return "R=" + Rlabels[p.value[0]] + " Å · seed " + seeds[p.value[1]] + "<br/>" +
            title + ": " + (isLog ? Number(raw).toExponential(2) : raw);
        } }),
      xAxis: xaxis("bond length R (Å)", { type: "category", data: Rlabels, nameGap: 26,
        splitArea: { show: true }, splitLine: { show: false }, axisLabel: { color: C.muted, fontSize: 10 } }),
      yAxis: yaxis("seed", { type: "category", data: seeds.map(String), nameGap: 24,
        splitArea: { show: true }, splitLine: { show: false } }),
      visualMap: {
        min: vmin, max: vmax, calculable: true, orient: "horizontal", left: "center", bottom: 6,
        itemWidth: 14, itemHeight: 120, inRange: { color: colors }, textStyle: { color: C.muted, fontSize: 10 },
        formatter: function (v) { return isLog ? Math.pow(10, v).toExponential(0) : v.toFixed(0); },
        dimension: 2,
      },
      series: [{
        type: "heatmap",
        data: data.map(function (d, i) { return [d[0], d[1], vals[i]]; }),
        itemStyle: { borderColor: "#ffffff", borderWidth: 2 },
        emphasis: { itemStyle: { borderColor: C.ink, borderWidth: 1 } },
      }],
    }));
  }

  // ---- gate counts vs paper bands ---------------------------------------
  make("chart-gates", base({
    legend: { type: "scroll", top: 10, left: "center", textStyle: { color: C.muted, fontSize: 11 },
      icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18, data: ["two-qubit (CX)", "total (excl. HF-X)"] },
    tooltip: tip({ trigger: "axis", axisPointer: { type: "shadow" } }),
    xAxis: xaxis("bond length R (Å)", { type: "category", data: Rlabels }),
    yAxis: yaxis("gate count", { type: "value" }),
    series: [
      { name: "two-qubit (CX)", type: "bar", itemStyle: { color: C.blue, borderRadius: [3, 3, 0, 0] },
        data: bonds.map(function (b) { return +b.two_qubit.mean.toFixed(1); }),
        markLine: { silent: true, symbol: "none", lineStyle: { color: C.blue, type: "dashed", opacity: 0.7 },
          label: { color: C.blue, fontSize: 10, position: "insideStartTop", formatter: "paper CX " + T.two_qubit_mean },
          data: [{ yAxis: T.two_qubit_mean }] } },
      { name: "total (excl. HF-X)", type: "bar", itemStyle: { color: C.violet, borderRadius: [3, 3, 0, 0] },
        data: bonds.map(function (b) { return +b.total.mean.toFixed(1); }),
        markLine: { silent: true, symbol: "none", lineStyle: { color: C.violet, type: "dashed", opacity: 0.7 },
          label: { color: C.violet, fontSize: 10, position: "insideEndTop", formatter: "paper total " + T.total_mean },
          data: [{ yAxis: T.total_mean }] } },
    ],
  }));

  // ---- convergence (interactive run picker) -----------------------------
  var convSel = document.getElementById("conv-select");
  var convNames = Object.keys(D.convergence);
  var convChart, convDimChart;
  if (convSel && convNames.length) {
    convNames.forEach(function (n) {
      var o = document.createElement("option"); o.value = n; o.textContent = n; convSel.appendChild(o);
    });
    convSel.addEventListener("change", function () { renderConv(convSel.value); });
    renderConv(convNames[0]);
  }
  function renderConv(name) {
    var r = D.convergence[name];
    if (!r) return;
    var it = r.iteration;
    var step = Math.max(1, Math.ceil(it.length / 10));
    if (!convChart) convChart = make("chart-conv", {});
    convChart.setOption(base({
      legend: { type: "scroll", top: 10, left: "center", textStyle: { color: C.muted, fontSize: 11 },
        icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18, data: ["best |error|", "mean loss"] },
      tooltip: tip({ trigger: "axis" }),
      grid: { left: 16, right: 22, top: 56, bottom: 16, containLabel: true },
      xAxis: xaxis("training iteration", { type: "category", data: it, boundaryGap: false,
        axisLabel: { color: C.muted, fontSize: 11, interval: step - 1 } }),
      yAxis: [
        yaxis("best |error| (Ha)", { type: "log", min: 1e-16, position: "left",
          axisLabel: { color: C.muted, fontSize: 11, formatter: decadeLabel },
          minorTick: { show: false }, minorSplitLine: { show: false } }),
        yaxis("mean loss", { type: "value", position: "right", splitLine: { show: false } }),
      ],
      series: [
        { name: "best |error|", type: "line", yAxisIndex: 0, smooth: true, symbol: "none",
          lineStyle: { color: C.blue, width: 2 }, areaStyle: { color: "rgba(43,108,176,0.07)" },
          data: r.best_error.map(function (v) { return Math.max(v, 1e-16); }) },
        { name: "mean loss", type: "line", yAxisIndex: 1, smooth: true, symbol: "none",
          lineStyle: { color: C.orange, width: 1.5 }, data: r.mean_loss },
      ],
    }), true);

    if (!convDimChart) convDimChart = make("chart-conv-dim", {});
    convDimChart.setOption(base({
      legend: { type: "scroll", top: 10, left: "center", textStyle: { color: C.muted, fontSize: 11 },
        icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18, data: ["subspace dim", "two-qubit gates"] },
      tooltip: tip({ trigger: "axis" }),
      grid: { left: 16, right: 22, top: 56, bottom: 16, containLabel: true },
      xAxis: xaxis("training iteration", { type: "category", data: it, boundaryGap: false,
        axisLabel: { color: C.muted, fontSize: 11, interval: step - 1 } }),
      yAxis: [
        yaxis("subspace dim", { type: "value", position: "left" }),
        yaxis("CX gates", { type: "value", position: "right", splitLine: { show: false } }),
      ],
      series: [
        { name: "subspace dim", type: "line", yAxisIndex: 0, smooth: true, symbol: "none",
          lineStyle: { color: C.green, width: 2 }, areaStyle: { color: "rgba(47,133,90,0.07)" }, data: r.subspace_dim },
        { name: "two-qubit gates", type: "line", yAxisIndex: 1, smooth: true, symbol: "none",
          lineStyle: { color: C.violet, width: 1.5 }, data: r.two_qubit },
      ],
    }), true);
  }

  // ---- model scaling ----------------------------------------------------
  var M = D.models, mlabels = M.map(function (m) { return m.label; });
  var mcolors = M.map(function (m) { return m.family === "gqe" ? C.orange : C.blue; });
  function modelBar(id, field, title, unit, log) {
    make(id, base({
      grid: { left: 16, right: 22, top: 30, bottom: 16, containLabel: true },
      legend: { show: false },
      tooltip: tip({ trigger: "axis", axisPointer: { type: "shadow" },
        valueFormatter: function (v) { return v == null ? "n/a" : v.toLocaleString() + " " + unit; } }),
      xAxis: xaxis("", { type: "category", data: mlabels,
        axisLabel: { interval: 0, fontSize: 11, color: C.muted, lineHeight: 14,
          formatter: function (v) { return v.replace(" (", "\n("); } } }),
      yAxis: yaxis(title, log ? { type: "log" } : { type: "value" }),
      series: [{
        type: "bar", barWidth: "50%",
        data: M.map(function (m, i) { return { value: m[field], itemStyle: { color: mcolors[i], borderRadius: [4, 4, 0, 0] } }; }),
        label: { show: true, position: "top", color: C.ink, fontSize: 11, fontFamily: "monospace",
          formatter: function (p) { var v = Number(p.value); return v >= 1000 ? abbr(v) : (v >= 1 ? v.toFixed(1) : v.toFixed(3)); } },
      }],
    }));
  }
  modelBar("chart-params", "parameters", "parameters (log)", "params", true);
  modelBar("chart-mem", "param_memory_mb", "parameter memory (MB)", "MB", false);
  modelBar("chart-uptime", "opt_update_mean_s", "update time (s)", "s", false);

  // ---- runtime breakdown ------------------------------------------------
  var rt = D.runtime;
  make("chart-runtime", base({
    grid: undefined,
    tooltip: tip({ trigger: "item", formatter: function (p) { return p.name + ": " + p.value.toFixed(3) + " s (" + p.percent + "%)"; } }),
    legend: { type: "scroll", bottom: 6, left: "center", textStyle: { color: C.muted, fontSize: 11 },
      icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 16 },
    series: [{
      type: "pie", radius: ["44%", "68%"], center: ["50%", "46%"], avoidLabelOverlap: true,
      itemStyle: { borderColor: "#ffffff", borderWidth: 2 },
      label: { color: C.ink, fontSize: 11, formatter: "{b}\n{d}%" },
      data: [
        { name: "evaluation", value: rt.evaluation, itemStyle: { color: C.blue } },
        { name: "optimization", value: rt.optimization, itemStyle: { color: C.violet } },
        { name: "generation", value: rt.generation, itemStyle: { color: C.green } },
      ],
    }],
  }));

  make("chart-job-runtime", base({
    grid: { left: 16, right: 22, top: 24, bottom: 16, containLabel: true },
    legend: { show: false },
    tooltip: tip({ trigger: "axis", axisPointer: { type: "shadow" },
      valueFormatter: function (v) { return v == null ? "n/a" : v.toFixed(1) + " s"; } }),
    xAxis: xaxis("bond length R (Å)", { type: "category", data: Rlabels }),
    yAxis: yaxis("mean run time (s)", { type: "value" }),
    series: [{ type: "bar", itemStyle: { color: C.orange, borderRadius: [3, 3, 0, 0] },
      data: bonds.map(function (b) { return +b.runtime_s.mean.toFixed(1); }) }],
  }));

  // ---- PES table --------------------------------------------------------
  var tbody = document.getElementById("pes-tbody");
  if (tbody) {
    bonds.forEach(function (b) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td class='mono'>" + b.R.toFixed(2) + "</td>" +
        "<td class='mono'>" + b.casci.toFixed(9) + "</td>" +
        "<td class='mono'>" + b.gqkae.mean.toFixed(9) + "</td>" +
        "<td class='mono'>" + b.abs_error.mean.toExponential(2) + "</td>" +
        "<td class='mono'>" + b.two_qubit.mean.toFixed(1) + "</td>" +
        "<td class='mono'>" + b.total.mean.toFixed(1) + "</td>" +
        "<td class='mono'>" + b.runtime_s.mean.toFixed(0) + "</td>" +
        "<td><span class='pill pass'>" + b.n + "/" + b.n + "</span></td>";
      tbody.appendChild(tr);
    });
  }

  // ---- TOC scroll-spy + resize ------------------------------------------
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  var sections = navLinks.map(function (a) { return document.querySelector(a.getAttribute("href")); }).filter(Boolean);
  var spy = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        navLinks.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id); });
      }
    });
  }, { rootMargin: "-15% 0px -75% 0px" });
  sections.forEach(function (s) { spy.observe(s); });

  var rztimer;
  window.addEventListener("resize", function () {
    clearTimeout(rztimer);
    rztimer = setTimeout(function () { charts.forEach(function (c) { c.resize(); }); }, 120);
  });
})();
