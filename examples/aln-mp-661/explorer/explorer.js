"use strict";

const $ = (id) => document.getElementById(id);
const initialQuery = new URLSearchParams(location.search).get("query");
const conversionToMicroCPerCm2 = 1602.176634;
const colors = {
  H: "#f6f7fb", C: "#35404b", N: "#4267d5", O: "#de5a55", F: "#57a85f",
  P: "#f28c36", S: "#e2be38", Cl: "#37a875", Ba: "#4f9c72", Ti: "#8795a3",
  Zr: "#688597", Pb: "#62606d", Bi: "#aa746e", Fe: "#d27443", Mn: "#8b68a8",
  Nb: "#758ba3", Ta: "#526b85", W: "#506372",
};
const radii = {
  H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05, Cl: 1.02,
  Ba: 2.15, Ti: 1.6, Zr: 1.75, Pb: 1.46, Bi: 1.48, Fe: 1.32, Mn: 1.39,
  Nb: 1.64, Ta: 1.7, W: 1.62,
};

let data;
let frameIndex = 0;
let selectedAtom = 0;
let yaw = 0.72;
let pitch = -0.42;
let zoom = 1;
let structureView = "3d";
let pan = { x: 0, y: 0 };
let playing = null;
let drag = null;
let chartDrag = null;
let lastProjectedAtoms = [];

function finite(value, fallback = NaN) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function fmt(value, digits = 3) {
  const number = finite(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function norm(vector) {
  return Math.hypot(...vector);
}

function cartesian(fractional, cell) {
  return [0, 1, 2].map((axis) =>
    fractional.reduce((sum, value, vector) => sum + value * cell[vector][axis], 0),
  );
}

function determinant(matrix) {
  return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

function inverse3(a) {
  const d = determinant(a);
  if (Math.abs(d) < 1e-12) throw new Error("cell matrix is singular");
  return [
    [(a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d, (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / d, (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d],
    [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d, (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / d, (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d],
    [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d, (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / d, (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d],
  ];
}

function multiply3(a, b) {
  return a.map((row) => b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));
}

function cellVolume(cell) {
  return Math.abs(determinant(cell));
}

function cartesianPolarization(frame) {
  const cell = frame.cell;
  const reduced = frame.reduced_polarization;
  const volume = cellVolume(cell);
  return [0, 1, 2].map((axis) =>
    cell.reduce((sum, vector, index) => sum + vector[axis] * reduced[index], 0)
      / volume * conversionToMicroCPerCm2,
  );
}

function cellParameters(cell) {
  const lengths = cell.map(norm);
  const angle = (u, v) => Math.acos(Math.max(-1, Math.min(1,
    u.reduce((sum, x, i) => sum + x * v[i], 0) / (norm(u) * norm(v)),
  ))) * 180 / Math.PI;
  return { lengths, angles: [angle(cell[1], cell[2]), angle(cell[0], cell[2]), angle(cell[0], cell[1])] };
}

function principalStretches(deformation) {
  const transpose = deformation[0].map((_, column) => deformation.map((row) => row[column]));
  const symmetric = multiply3(deformation, transpose);
  // Jacobi diagonalization of F F^T; eigenvalue square roots are principal stretches.
  for (let iteration = 0; iteration < 24; iteration += 1) {
    let p = 0;
    let q = 1;
    for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(symmetric[i][j]) > Math.abs(symmetric[p][q])) [p, q] = [i, j];
    }
    if (Math.abs(symmetric[p][q]) < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * symmetric[p][q], symmetric[q][q] - symmetric[p][p]);
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const app = symmetric[p][p];
    const aqq = symmetric[q][q];
    const apq = symmetric[p][q];
    symmetric[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    symmetric[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    symmetric[p][q] = symmetric[q][p] = 0;
    for (const k of [0, 1, 2]) {
      if (k === p || k === q) continue;
      const akp = symmetric[k][p];
      const akq = symmetric[k][q];
      symmetric[k][p] = symmetric[p][k] = c * akp - s * akq;
      symmetric[k][q] = symmetric[q][k] = s * akp + c * akq;
    }
  }
  return [0, 1, 2].map((i) => Math.sqrt(Math.max(0, symmetric[i][i]))).sort((a, b) => a - b);
}

function distortions(frame) {
  const initial = data.frames[0];
  const deformation = multiply3(inverse3(initial.cell), frame.cell);
  let strainSquared = 0;
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      strainSquared += (deformation[i][j] - (i === j ? 1 : 0)) ** 2;
    }
  }
  const displacementSquared = frame.fractional_positions.reduce((sum, position, atom) => {
    const delta = position.map((value, axis) => {
      let difference = value - initial.fractional_positions[atom][axis];
      difference -= Math.round(difference);
      return difference;
    });
    return sum + norm(cartesian(delta, frame.cell)) ** 2;
  }, 0);
  const stretches = principalStretches(deformation);
  return {
    atomRmsA: Math.sqrt(displacementSquared / Math.max(1, frame.fractional_positions.length)),
    rmsStrainPercent: 100 * Math.sqrt(strainSquared / 3),
    maximumStretchDeviationPercent: 100 * Math.max(...stretches.map((x) => Math.abs(x - 1))),
    volumeChangePercent: 100 * (cellVolume(frame.cell) / cellVolume(initial.cell) - 1),
    stretches,
  };
}

function quantumMatrix(frame) {
  const volume = cellVolume(frame.cell);
  return [0, 1, 2].map((cartAxis) =>
    [0, 1, 2].map((vector) => frame.cell[vector][cartAxis] / volume * conversionToMicroCPerCm2),
  );
}

function resizeCanvas(canvas) {
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.height = Math.max(1, Math.round(bounds.height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: bounds.width, height: bounds.height };
}

function tooltipAt(event, index) {
  const frame = data.frames[index];
  const energies = data.frames.map((item) =>
    (item.energy_eV - data.frames[0].energy_eV) / Math.max(1, data.atom_count) * 1000,
  );
  const barrier = Math.max(...energies);
  const polarization = cartesianPolarization(frame);
  const reduced = frame.reduced_polarization.map((value) => fmt(value, 4));
  const tooltip = $("chart-tooltip");
  tooltip.className = "sample-tooltip";
  tooltip.textContent = [
    `Image ${index + 1} / ${data.frames.length}  ·  branch ${fmt(frame.coordinate, 3)}`,
    `Energy above +P: ${fmt(energies[index], 2)} meV/atom`,
    `Maximum sampled rise from +P: ${fmt(barrier, 2)} meV/atom`,
    `P = (${polarization.map((value) => fmt(value, 2)).join(", ")}) µC/cm²`,
    `Reduced p = (${reduced.join(", ")}) quanta`,
  ].join("\n");
  tooltip.style.display = "block";
  tooltip.style.left = `${Math.min(window.innerWidth - 345, event.clientX + 12)}px`;
  tooltip.style.top = `${Math.min(window.innerHeight - 160, event.clientY + 12)}px`;
}

function chartImageIndex(canvas, event) {
  const bounds = canvas.getBoundingClientRect();
  const chartLeft = finite(canvas.dataset.chartLeft, 52);
  const chartWidth = finite(canvas.dataset.chartWidth, bounds.width - 64);
  const x = event.clientX - bounds.left;
  return Math.round(Math.max(0, Math.min(data.frames.length - 1,
    (x - chartLeft) / chartWidth * (data.frames.length - 1),
  )));
}

function scrubFromChart(canvas, event) {
  const nextIndex = chartImageIndex(canvas, event);
  if (nextIndex !== frameIndex) {
    frameIndex = nextIndex;
    draw();
  }
  tooltipAt(event, frameIndex);
}

function branchChart(canvas, title, series, selectedIndex) {
  const { context: c, width, height } = resizeCanvas(canvas);
  const left = 52;
  const right = 12;
  const top = 24;
  const bottom = 27;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const values = series.flatMap((item) => item.values.filter(Number.isFinite));
  if (!values.length) return;
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (Math.abs(high - low) < 1e-9) high = low + 1;
  const padding = (high - low) * 0.12;
  low -= padding;
  high += padding;
  c.clearRect(0, 0, width, height);
  c.font = "11px system-ui";
  c.fillStyle = "#294b60";
  c.fillText(title, 7, 15);
  c.strokeStyle = "#e0e8ed";
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = top + plotHeight * tick / 4;
    const value = high - (high - low) * tick / 4;
    c.beginPath();
    c.moveTo(left, y);
    c.lineTo(width - right, y);
    c.stroke();
    c.fillStyle = "#718391";
    c.textAlign = "right";
    c.fillText(fmt(value, title.startsWith("Energy") ? 0 : 1), left - 5, y + 4);
  }
  c.textAlign = "left";
  const xFor = (index) => left + plotWidth * (data.frames.length < 2 ? 0 : index / (data.frames.length - 1));
  for (const item of series) {
    c.strokeStyle = item.color;
    c.lineWidth = 2;
    c.beginPath();
    item.values.forEach((value, index) => {
      const x = xFor(index);
      const y = top + plotHeight * (high - value) / (high - low);
      index ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
  }
  const markerX = xFor(selectedIndex);
  c.strokeStyle = "#263e50";
  c.setLineDash([4, 4]);
  c.beginPath();
  c.moveTo(markerX, top);
  c.lineTo(markerX, top + plotHeight);
  c.stroke();
  c.setLineDash([]);
  c.fillStyle = "#70818d";
  c.textAlign = "center";
  c.fillText("+P", left, top + plotHeight + 19);
  c.fillText("N", left + plotWidth / 2, top + plotHeight + 19);
  c.fillText("−P", left + plotWidth, top + plotHeight + 19);
  canvas.dataset.chartLeft = String(left);
  canvas.dataset.chartWidth = String(plotWidth);
  if (!canvas.dataset.eventsReady) {
    canvas.dataset.eventsReady = "true";
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      chartDrag = { canvas, pointerId: event.pointerId };
      canvas.setPointerCapture(event.pointerId);
      scrubFromChart(canvas, event);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (chartDrag?.canvas === canvas && chartDrag.pointerId === event.pointerId) {
        scrubFromChart(canvas, event);
      } else {
        tooltipAt(event, chartImageIndex(canvas, event));
      }
    });
    const endChartDrag = (event) => {
      if (chartDrag?.canvas === canvas && chartDrag.pointerId === event.pointerId) {
        chartDrag = null;
      }
    };
    canvas.addEventListener("pointerup", endChartDrag);
    canvas.addEventListener("pointercancel", endChartDrag);
    canvas.addEventListener("lostpointercapture", endChartDrag);
    canvas.addEventListener("pointerleave", () => {
      if (chartDrag?.canvas !== canvas) $("chart-tooltip").style.display = "none";
    });
    canvas.addEventListener("click", (event) => {
      frameIndex = chartImageIndex(canvas, event);
      draw();
    });
  }
}

function project(point, cell, width, height) {
  const center = cartesian([0.5, 0.5, 0.5], cell);
  const vector = point.map((value, i) => value - center[i]);
  if (structureView !== "3d") {
    const axes = {
      x: [1, 2, 0], // look along x; show the yz plane
      y: [0, 2, 1], // look along y; show the xz plane
      z: [0, 1, 2], // look along z; show the xy plane
    }[structureView];
    const [horizontalAxis, verticalAxis, depthAxis] = axes;
    const horizontalExtent = cell.reduce(
      (sum, basis) => sum + Math.abs(basis[horizontalAxis]), 0,
    );
    const verticalExtent = cell.reduce(
      (sum, basis) => sum + Math.abs(basis[verticalAxis]), 0,
    );
    const scale = Math.min(
      width * 0.76 / Math.max(horizontalExtent, 1e-8),
      height * 0.72 / Math.max(verticalExtent, 1e-8),
    ) * zoom;
    return {
      x: width / 2 + pan.x + vector[horizontalAxis] * scale,
      y: height / 2 + pan.y - vector[verticalAxis] * scale,
      z: vector[depthAxis],
      scale,
    };
  }
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const x = cy * vector[0] - sy * vector[1];
  const y0 = sy * vector[0] + cy * vector[1];
  const y = cp * y0 - sp * vector[2];
  const z = sp * y0 + cp * vector[2];
  const extent = Math.max(...cell.map(norm), 1);
  const scale = Math.min(width, height) * 0.68 / extent * zoom;
  return {
    x: width / 2 + pan.x + x * scale,
    y: height / 2 + pan.y - y * scale,
    z,
    scale,
  };
}

function drawStructure() {
  if (!data) return;
  const canvas = $("structure");
  const { context: c, width, height } = resizeCanvas(canvas);
  const frame = data.frames[frameIndex];
  const { cell, fractional_positions: fractions, symbols } = frame;
  c.clearRect(0, 0, width, height);
  const corners = Array.from({ length: 8 }, (_, index) =>
    project(cartesian([index & 1, (index >> 1) & 1, (index >> 2) & 1], cell), cell, width, height),
  );
  if ($("cell").checked) {
    c.strokeStyle = "#9babb6";
    c.lineWidth = 1.15;
    for (let index = 0; index < 8; index += 1) {
      for (const bit of [1, 2, 4]) {
        if (index & bit) continue;
        c.beginPath();
        c.moveTo(corners[index].x, corners[index].y);
        c.lineTo(corners[index | bit].x, corners[index | bit].y);
        c.stroke();
      }
    }
  }
  const points = fractions.map((fractional) =>
    project(cartesian(fractional, cell), cell, width, height),
  );
  lastProjectedAtoms = points;
  if ($("motion").checked && data.frames.length > 1) {
    const base = data.frames[0].fractional_positions;
    c.save();
    c.setLineDash([3, 3]);
    for (let atom = 0; atom < fractions.length; atom += 1) {
      const delta = fractions[atom].map((value, axis) => {
        let difference = value - base[atom][axis];
        difference -= Math.round(difference);
        return difference;
      });
      if (norm(delta) < 1e-4) continue;
      const start = fractions[atom].map((value, axis) => value - delta[axis]);
      const projectedStart = project(cartesian(start, cell), cell, width, height);
      c.strokeStyle = "#344f6380";
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(projectedStart.x, projectedStart.y);
      c.lineTo(points[atom].x, points[atom].y);
      c.stroke();
    }
    c.restore();
  }
  if ($("bonds").checked) {
    c.strokeStyle = "#51667599";
    c.lineWidth = 1.5;
    for (let i = 0; i < symbols.length; i += 1) {
      for (let j = i + 1; j < symbols.length; j += 1) {
        const delta = fractions[j].map((value, axis) => {
          let difference = value - fractions[i][axis];
          difference -= Math.round(difference);
          return difference;
        });
        const distance = norm(cartesian(delta, cell));
        const cutoff = 1.18 * ((radii[symbols[i]] || 0.9) + (radii[symbols[j]] || 0.9)) + 0.1;
        if (distance <= 0.15 || distance >= cutoff) continue;
        const end = fractions[i].map((value, axis) => value + delta[axis]);
        const projectedEnd = project(cartesian(end, cell), cell, width, height);
        c.beginPath();
        c.moveTo(points[i].x, points[i].y);
        c.lineTo(projectedEnd.x, projectedEnd.y);
        c.stroke();
      }
    }
  }
  const atomScale = finite($("atom-size").value, 2);
  const ordered = points.map((point, index) => ({ point, index })).sort((a, b) => a.point.z - b.point.z);
  for (const { point, index } of ordered) {
    const radius = Math.max(3, Math.min(18,
      (radii[symbols[index]] || 0.9) * 0.18 * point.scale * atomScale,
    ));
    const gradient = c.createRadialGradient(point.x - radius * 0.32, point.y - radius * 0.38, radius * 0.08, point.x, point.y, radius);
    const color = colors[symbols[index]] || "#8295a3";
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(0.28, color);
    gradient.addColorStop(1, "#1c2d3a");
    c.fillStyle = gradient;
    c.beginPath();
    c.arc(point.x, point.y, radius, 0, 2 * Math.PI);
    c.fill();
    if (index === selectedAtom) {
      c.strokeStyle = "#f1a72f";
      c.lineWidth = 2.5;
      c.beginPath();
      c.arc(point.x, point.y, radius + 4, 0, 2 * Math.PI);
      c.stroke();
    }
    if ($("labels").checked) {
      c.fillStyle = "#172b3b";
      c.font = "10px system-ui";
      c.textAlign = "center";
      c.fillText(symbols[index], point.x, point.y - radius - 3);
    }
  }
  const unique = [...new Set(symbols)];
  $("legend").replaceChildren(...unique.map((symbol) => {
    const item = document.createElement("span");
    item.textContent = symbol;
    item.style.background = colors[symbol] || "#8295a3";
    return item;
  }));
  const branchName = frame.coordinate > 0 ? "+P → N" : frame.coordinate < 0 ? "N → −P" : "nonpolar candidate";
  const viewLabels = {
    "3d": "3D perspective",
    x: "along x · yz projection",
    y: "along y · xz projection",
    z: "along z · xy projection",
  };
  $("caption").textContent = `${branchName} · ${viewLabels[structureView]} · image ${frameIndex + 1}/${data.frames.length} · selected ${symbols[selectedAtom]} ${selectedAtom + 1}`;
  const parameters = cellParameters(cell);
  $("metrics").textContent = `Cell ${parameters.lengths.map((x) => fmt(x, 2)).join(" × ")} Å · volume ${fmt(cellVolume(cell), 1)} Å³ · angles ${parameters.angles.map((x) => fmt(x, 1)).join("° / ")}°`;
}

function setStructureView(view) {
  structureView = view;
  pan = { x: 0, y: 0 };
  zoom = 1;
  const descriptions = {
    "3d": "Drag to rotate · Shift-drag to pan",
    x: "Along x · y–z projection · drag to pan",
    y: "Along y · x–z projection · drag to pan",
    z: "Along z · x–y projection · drag to pan",
  };
  document.querySelectorAll("[data-structure-view]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.structureView === view));
  });
  $("structure-hint").textContent = `${descriptions[view]} · scroll to zoom · click an atom to inspect its Born effective charge.`;
  drawStructure();
}

function tensorTable(matrix, rowLabels = ["x", "y", "z"], columnLabels = ["x", "y", "z"], digits = 4) {
  if (!Array.isArray(matrix) || matrix.length !== 3) return "<p class=muted>Not available</p>";
  const header = `<tr><th></th>${columnLabels.map((x) => `<th>${x}</th>`).join("")}</tr>`;
  const rows = matrix.map((row, index) =>
    `<tr><th>${rowLabels[index]}</th>${row.map((value) => `<td>${fmt(value, digits)}</td>`).join("")}</tr>`,
  ).join("");
  return `<table>${header}${rows}</table>`;
}

function renderReport() {
  const frame = data.frames[frameIndex];
  const parameters = cellParameters(frame.cell);
  const distortion = distortions(frame);
  const polarization = cartesianPolarization(frame);
  const quantum = quantumMatrix(frame);
  const energyRise = (frame.energy_eV - data.frames[0].energy_eV) / Math.max(1, data.atom_count) * 1000;
  const barrier = Math.max(...data.frames.map((item) =>
    (item.energy_eV - data.frames[0].energy_eV) / Math.max(1, data.atom_count) * 1000,
  ));
  const symbol = frame.symbols[selectedAtom];
  const becs = frame.becs_e;
  const selectedBec = becs && becs[selectedAtom];
  const allBecs = becs
    ? `<details><summary>All atomic Born effective charges (${becs.length} atoms)</summary><table><tr><th>Atom</th><th>Symbol</th><th>Zxx</th><th>Zxy</th><th>Zxz</th><th>Zyx</th><th>Zyy</th><th>Zyz</th><th>Zzx</th><th>Zzy</th><th>Zzz</th></tr>${becs.map((tensor, atom) => `<tr><td>${atom + 1}</td><td>${frame.symbols[atom]}</td>${tensor.flat().map((value) => `<td>${fmt(value, 4)}</td>`).join("")}</tr>`).join("")}</table></details>`
    : "<p class=muted>Born charges were not evaluated for this image.</p>";
  const axes = ["x", "y", "z"];
  const qTable = `<table><tr><th>Cartesian</th><th>qₐ</th><th>qᵦ</th><th>q𝚌</th></tr>${quantum.map((row, i) => `<tr><th>${axes[i]}</th>${row.map((value) => `<td>${fmt(value, 3)}</td>`).join("")}</tr>`).join("")}</table>`;
  const atomOptions = frame.symbols.map((item, index) =>
    `<option value="${index}" ${index === selectedAtom ? "selected" : ""}>Atom ${index + 1} · ${item}</option>`,
  ).join("");
  $("details").innerHTML = `
    <div class="report-head"><div><h2>Per-image properties</h2><p class="hint">Image ${frameIndex + 1}/${data.frames.length} · branch coordinate ${fmt(frame.coordinate, 3)} · values for the selected structure.</p></div><label class="hint">Selected atom <select id="atom-select" class="atom-select">${atomOptions}</select></label></div>
    <div class="report-grid">
      <section class="report-block"><h3>Cell and distortion from +P endpoint</h3>
        <table><tr><th>Cell lengths (Å)</th><td>${parameters.lengths.map((x) => fmt(x, 4)).join(" · ")}</td></tr><tr><th>Cell angles (°)</th><td>${parameters.angles.map((x) => fmt(x, 3)).join(" · ")}</td></tr><tr><th>Volume (Å³)</th><td>${fmt(cellVolume(frame.cell), 4)}</td></tr><tr><th>Atomic RMS displacement (Å)</th><td>${fmt(distortion.atomRmsA, 4)}</td></tr><tr><th>Cell RMS deformation (%)</th><td>${fmt(distortion.rmsStrainPercent, 3)}</td></tr><tr><th>Maximum principal stretch deviation (%)</th><td>${fmt(distortion.maximumStretchDeviationPercent, 3)}</td></tr><tr><th>Volume change (%)</th><td>${fmt(distortion.volumeChangePercent, 3)}</td></tr></table>
      </section>
      <section class="report-block"><h3>Polarization</h3>
        <p class="pol-vector">P = (${polarization.map((x) => fmt(x, 3)).join(", ")}) µC/cm²<br>|P| = ${fmt(norm(polarization), 3)} µC/cm²</p>
        <p>Reduced p = (${frame.reduced_polarization.map((x) => fmt(x, 5)).join(", ")}) quantum units</p>
        <h3>Polarization quantum matrix</h3><p class="hint">Columns are qₐ, qᵦ, q𝚌; entries are Cartesian µC/cm² per quantum.</p>${qTable}
      </section>
      <section class="report-block"><h3>Response tensors</h3>
        <p class="hint">Born effective charge Z* for atom ${selectedAtom + 1} (${symbol}); tensor components in e.</p>${tensorTable(selectedBec)}
        <h3>Polarizability tensor</h3><p class="hint">MACE-Field units: e/(V Å).</p>${tensorTable(frame.polarizability_e_per_V_A, axes, axes, 5)}
        ${allBecs}
      </section>
    </div>
    <p class="barrier-note">Image energy relative to +P: ${fmt(energyRise, 3)} meV/atom. Maximum sampled energy rise over this geometric path: ${fmt(barrier, 3)} meV/atom; this is not a relaxed activation barrier.</p>`;
  $("atom-select").addEventListener("change", (event) => {
    selectedAtom = Number(event.target.value);
    draw();
  });
}

function setSummary() {
  const parent = data.parent || {};
  const validation = data.validation || {};
  const items = [
    ["Formula", data.formula], ["Atoms", data.atom_count],
    ["Parent candidate", parent.candidate_id || "—"],
    ["Parent symmetry", parent.pointgroup || "—"],
  ];
  if (validation.polar_mpid) items.push(["Validated polar ID", validation.polar_mpid]);
  if (validation.nonpolar_mpid) items.push(["Validated parent ID", validation.nonpolar_mpid]);
  if (validation.rank1_rmsd_A !== undefined) {
    items.push(["Rank 1 atom / cell RMSD", `${fmt(validation.rank1_rmsd_A)} / ${fmt(validation.rank1_cell_rmsd_A)} Å`]);
  }
  if (validation.path_dtw_rmsd_A !== undefined) items.push(["Path DTW RMSD", `${fmt(validation.path_dtw_rmsd_A)} Å`]);
  if (validation.branch_rmse !== undefined) items.push(["Branch RMSE mod quantum", fmt(validation.branch_rmse, 4)]);
  $("summary").replaceChildren(...items.map(([label, value]) => {
    const item = document.createElement("div");
    const caption = document.createElement("span");
    const content = document.createElement("strong");
    caption.textContent = label;
    content.textContent = value ?? "—";
    item.append(caption, content);
    return item;
  }));
  $("title").textContent = `${data.formula || data.query_id} · P–N–−P branch`;
  selectedAtom = Math.min(selectedAtom, Math.max(0, data.atom_count - 1));
}

function draw() {
  if (!data) return;
  const energies = data.frames.map((frame) =>
    (frame.energy_eV - data.frames[0].energy_eV) / Math.max(1, data.atom_count) * 1000,
  );
  const cartesianP = data.frames.map(cartesianPolarization);
  branchChart($("energy"), "Energy above +P (meV/atom)", [{ values: energies, color: "#d87942" }], frameIndex);
  branchChart($("polarization"), "Cartesian P (µC/cm²) · x red · y blue · z green", [
    { values: cartesianP.map((value) => value[0]), color: "#d95f68" },
    { values: cartesianP.map((value) => value[1]), color: "#377eb8" },
    { values: cartesianP.map((value) => value[2]), color: "#338c73" },
  ], frameIndex);
  drawStructure();
  renderReport();
  $("frame").max = String(Math.max(0, data.frames.length - 1));
  $("frame").value = String(frameIndex);
  $("frame-label").textContent = `${frameIndex + 1}/${data.frames.length}`;
}

function xyz() {
  const frame = data.frames[frameIndex];
  const lines = [String(frame.symbols.length), `Lattice="${frame.cell.flat().join(" ")}" Properties=species:S:1:pos:R:3 pbc="T T T"`];
  frame.symbols.forEach((symbol, atom) => {
    lines.push(`${symbol} ${cartesian(frame.fractional_positions[atom], frame.cell).map((x) => x.toFixed(8)).join(" ")}`);
  });
  return `${lines.join("\n")}\n`;
}

async function loadQuery(id) {
  const response = await fetch(`data/${encodeURIComponent(id)}.json`);
  if (!response.ok) throw new Error(`data fetch failed (${response.status})`);
  data = await response.json();
  frameIndex = 0;
  selectedAtom = 0;
  const params = new URLSearchParams(location.search);
  params.set("query", id);
  history.replaceState(null, "", `${location.pathname}?${params.toString()}`);
  setSummary();
  draw();
}

async function load() {
  try {
    const manifest = await fetch("manifest.json").then((response) => {
      if (!response.ok) throw new Error(`manifest fetch failed (${response.status})`);
      return response.json();
    });
    const ids = manifest.queries.map((item) => item.query_id);
    if (!ids.length) {
      document.querySelector("main").innerHTML = "<p class=error>No sampled P-N--P branches were produced for these inputs.</p>";
      return;
    }
    const selectorLabel = $("material-selector");
    const selector = $("material");
    selectorLabel.hidden = ids.length < 2;
    selector.replaceChildren(...manifest.queries.map((item) => {
      const option = document.createElement("option");
      option.value = item.query_id;
      option.textContent = `${item.name || item.query_id} · ${item.formula || ""}`;
      return option;
    }));
    selector.addEventListener("change", () => {
      loadQuery(selector.value).catch((error) => {
        document.querySelector("main").innerHTML = `<p class="error">Could not load recovery data: ${error.message}.</p>`;
      });
    });
    const id = ids.includes(initialQuery) ? initialQuery : ids[0];
    selector.value = id;
    await loadQuery(id);
  } catch (error) {
    document.querySelector("main").innerHTML = `<p class="error">Could not load recovery data: ${error.message}. Serve this directory locally with <code>python -m http.server</code>.</p>`;
  }
}

$("frame").addEventListener("input", (event) => { frameIndex = Number(event.target.value); draw(); });
["cell", "bonds", "motion", "labels"].forEach((id) => $(id).addEventListener("change", drawStructure));
$("atom-size").addEventListener("input", (event) => {
  $("atom-size-value").textContent = `${Number(event.target.value).toFixed(2)}×`;
  drawStructure();
});
$("play").addEventListener("click", () => {
  if (playing) {
    clearInterval(playing);
    playing = null;
    $("play").textContent = "Play";
    return;
  }
  $("play").textContent = "Pause";
  playing = setInterval(() => { frameIndex = (frameIndex + 1) % data.frames.length; draw(); }, 260);
});
$("reset").addEventListener("click", () => {
  yaw = 0.72;
  pitch = -0.42;
  setStructureView("3d");
});
document.querySelectorAll("[data-structure-view]").forEach((button) => {
  button.addEventListener("click", () => setStructureView(button.dataset.structureView));
});
$("download").addEventListener("click", () => {
  const anchor = document.createElement("a");
  const blob = new Blob([xyz()], { type: "chemical/x-xyz" });
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${data.query_id}_image_${frameIndex + 1}.extxyz`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
});
const structureCanvas = $("structure");
structureCanvas.addEventListener("pointerdown", (event) => {
  drag = {
    x: event.clientX,
    y: event.clientY,
    moved: false,
    pan: structureView !== "3d" || event.shiftKey,
  };
  structureCanvas.setPointerCapture(event.pointerId);
});
structureCanvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
  if (drag.pan) {
    pan.x += dx;
    pan.y += dy;
  } else {
    yaw += dx * 0.009;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.009));
  }
  drag.x = event.clientX;
  drag.y = event.clientY;
  drawStructure();
});
structureCanvas.addEventListener("pointerup", (event) => {
  if (drag && !drag.moved && lastProjectedAtoms.length) {
    const bounds = structureCanvas.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    let closest = -1;
    let distance = 20;
    lastProjectedAtoms.forEach((point, index) => {
      const current = Math.hypot(point.x - x, point.y - y);
      if (current < distance) { closest = index; distance = current; }
    });
    if (closest >= 0) { selectedAtom = closest; draw(); }
  }
  drag = null;
});
structureCanvas.addEventListener("pointercancel", () => { drag = null; });
structureCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  zoom = Math.max(0.55, Math.min(2.6, zoom * Math.exp(-event.deltaY * 0.001)));
  drawStructure();
}, { passive: false });
window.addEventListener("resize", draw);
load();
