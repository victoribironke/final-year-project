/* ═══════════════════════════════════════════════════════════
   DemandCast – Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const API = ""; // Same-origin
let commodities = [];
let commodityMap = {};

// ── Init ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadCommodities();
  setupNavigation();
  setupForm();
  setupSliders();
  setupModelTabs();
  setDefaultMonth();
});

// ── Navigation ─────────────────────────────────────────────
function setupNavigation() {
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const section = link.dataset.section;
      showSection(section);
    });
  });
}

function showSection(sectionId) {
  document
    .querySelectorAll(".section")
    .forEach((s) => s.classList.remove("active"));
  document
    .querySelectorAll(".nav-link")
    .forEach((l) => l.classList.remove("active"));
  const section = document.getElementById(sectionId);
  if (section) section.classList.add("active");
  const link = document.querySelector(`.nav-link[data-section="${sectionId}"]`);
  if (link) link.classList.add("active");
}

// ── Load Commodities ───────────────────────────────────────
async function loadCommodities() {
  try {
    const res = await fetch(`${API}/api/commodities`);
    commodities = await res.json();
    commodityMap = {};
    commodities.forEach((c) => {
      commodityMap[c.name] = c;
    });
    populateCommodityDropdown();
    populateCommodityGrid();
  } catch (err) {
    console.error("Failed to load commodities:", err);
  }
}

function populateCommodityDropdown() {
  const select = document.getElementById("commodity");
  select.innerHTML = '<option value="">Select a commodity…</option>';
  commodities.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent = c.name;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => {
    const name = select.value;
    const info = document.getElementById("commodity-info");
    if (name && commodityMap[name]) {
      updateCommodityInfo(commodityMap[name]);
      info.classList.remove("hidden");
    } else {
      info.classList.add("hidden");
    }
  });
}

function updateCommodityInfo(c) {
  document.getElementById("info-commodity-name").textContent = c.name;
  document.getElementById("info-champion").textContent =
    `Champion: ${c.champion}`;
  document.getElementById("info-records").textContent = c.records || "—";
  document.getElementById("info-date-range").textContent = c.date_range
    ? `${c.date_range[0]} → ${c.date_range[1]}`
    : "—";
  document.getElementById("info-avg-demand").textContent =
    c.avg_demand != null ? c.avg_demand.toFixed(1) : "—";
  document.getElementById("info-avg-temp").textContent =
    c.avg_temp != null ? `${c.avg_temp.toFixed(1)}°C` : "—";

  // Update default temp/rainfall from commodity's historical averages
  if (c.avg_temp != null) {
    document.getElementById("temperature").value = c.avg_temp.toFixed(1);
    document.getElementById("temp-slider").value = c.avg_temp;
  }
  if (c.avg_rainfall != null) {
    document.getElementById("rainfall").value = Math.round(c.avg_rainfall);
    document.getElementById("rain-slider").value = Math.round(c.avg_rainfall);
  }
}

function populateCommodityGrid() {
  const grid = document.getElementById("commodity-grid");
  grid.innerHTML = "";
  commodities.forEach((c) => {
    const tile = document.createElement("div");
    tile.className = "commodity-tile";
    tile.innerHTML = `
      <div class="tile-name">${c.name}</div>
      <div class="tile-stats">
        <span class="tile-stat"><strong>${c.records || "—"}</strong> records</span>
        <span class="tile-stat">Avg demand: <strong>${c.avg_demand != null ? c.avg_demand.toFixed(1) : "—"}</strong></span>
      </div>
      <div class="tile-champion">🏆 ${c.champion}</div>
    `;
    tile.addEventListener("click", () => {
      // Navigate to predict with this commodity selected
      showSection("predict");
      document.getElementById("commodity").value = c.name;
      updateCommodityInfo(c);
      document.getElementById("commodity-info").classList.remove("hidden");
      document.getElementById("commodity").focus();
    });
    grid.appendChild(tile);
  });

  // Search functionality
  const searchInput = document.getElementById("explore-search");
  searchInput.addEventListener("input", () => {
    const q = searchInput.value.toLowerCase();
    document.querySelectorAll(".commodity-tile").forEach((tile) => {
      const name = tile.querySelector(".tile-name").textContent.toLowerCase();
      tile.style.display = name.includes(q) ? "" : "none";
    });
  });
}

// ── Form & Sliders ─────────────────────────────────────────
function setupSliders() {
  const tempSlider = document.getElementById("temp-slider");
  const tempInput = document.getElementById("temperature");
  const rainSlider = document.getElementById("rain-slider");
  const rainInput = document.getElementById("rainfall");

  tempSlider.addEventListener("input", () => {
    tempInput.value = tempSlider.value;
  });
  tempInput.addEventListener("input", () => {
    tempSlider.value = tempInput.value;
  });
  rainSlider.addEventListener("input", () => {
    rainInput.value = rainSlider.value;
  });
  rainInput.addEventListener("input", () => {
    rainSlider.value = rainInput.value;
  });
}

function setDefaultMonth() {
  const now = new Date();
  document.getElementById("month").value = now.getMonth() + 1;
  document.getElementById("year").value = now.getFullYear();
}

function setupForm() {
  document
    .getElementById("predict-form")
    .addEventListener("submit", async (e) => {
      e.preventDefault();
      await makePrediction();
    });
}

// ── Prediction ─────────────────────────────────────────────
async function makePrediction() {
  const btn = document.getElementById("predict-btn");
  const commodity = document.getElementById("commodity").value;
  if (!commodity) {
    shakeElement(document.getElementById("commodity"));
    return;
  }

  btn.classList.add("loading");
  btn.disabled = true;

  const body = {
    commodity,
    temperature: parseFloat(document.getElementById("temperature").value),
    rainfall: parseFloat(document.getElementById("rainfall").value),
    month: parseInt(document.getElementById("month").value),
    year: parseInt(document.getElementById("year").value),
    is_holiday: document.getElementById("is-holiday").checked,
  };

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Prediction failed");
    }

    const data = await res.json();
    displayResult(data, body);
  } catch (err) {
    console.error("Prediction error:", err);
    alert("Prediction failed: " + err.message);
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
}

function displayResult(data, inputs) {
  // Hide placeholder, show content
  document.getElementById("result-placeholder").classList.add("hidden");
  const content = document.getElementById("result-content");
  content.classList.remove("hidden");

  // Model badge
  document.getElementById("result-model-badge").textContent =
    data.champion_model;

  // Date
  const monthNames = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  document.getElementById("result-date").textContent =
    `${monthNames[inputs.month - 1]} ${inputs.year}`;

  // Commodity
  document.getElementById("result-commodity").textContent = data.commodity;

  // Demand number (animate)
  const demandEl = document.getElementById("result-demand");
  animateNumber(demandEl, data.predicted_demand);

  // Gauge
  const pct = Math.max(0, Math.min(100, data.predicted_demand));
  setTimeout(() => {
    document.getElementById("gauge-fill").style.width = pct + "%";
  }, 100);

  // Update gradient colour based on demand level
  if (pct > 70) {
    demandEl.style.background = "linear-gradient(135deg, #fb923c, #f472b6)";
  } else if (pct > 40) {
    demandEl.style.background = "linear-gradient(135deg, #22d3ee, #8b5cf6)";
  } else {
    demandEl.style.background = "linear-gradient(135deg, #34d399, #22d3ee)";
  }
  demandEl.style.webkitBackgroundClip = "text";
  demandEl.style.webkitTextFillColor = "transparent";
  demandEl.style.backgroundClip = "text";

  // Details
  document.getElementById("detail-temp").textContent =
    `${inputs.temperature}°C`;
  document.getElementById("detail-rain").textContent = `${inputs.rainfall}mm`;

  const isWet = inputs.month >= 4 && inputs.month <= 10;
  document.getElementById("detail-season").textContent = isWet
    ? "Wet Season"
    : "Dry Season";
  document.getElementById("detail-season-icon").textContent = isWet
    ? "🌧️"
    : "☀️";

  document.getElementById("detail-holiday").textContent = inputs.is_holiday
    ? "Yes"
    : "No";
  document.getElementById("detail-holiday-icon").textContent = inputs.is_holiday
    ? "🎉"
    : "📅";

  // Scroll result into view on mobile
  if (window.innerWidth < 768) {
    document
      .getElementById("result-card")
      .scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// ── Animate number ─────────────────────────────────────────
function animateNumber(el, target) {
  const duration = 800;
  const start = performance.now();
  const from = 0;

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (target - from) * eased;
    el.textContent = current.toFixed(2);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Shake animation ────────────────────────────────────────
function shakeElement(el) {
  el.style.animation = "none";
  el.offsetHeight; // reflow
  el.style.animation = "shake 0.4s ease";
  el.style.borderColor = "#ef4444";
  setTimeout(() => {
    el.style.borderColor = "";
  }, 1500);
}

// Add shake keyframes dynamically
const style = document.createElement("style");
style.textContent = `
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    20% { transform: translateX(-6px); }
    40% { transform: translateX(6px); }
    60% { transform: translateX(-4px); }
    80% { transform: translateX(4px); }
  }
`;
document.head.appendChild(style);

// ── Model Tabs ─────────────────────────────────────────────
function setupModelTabs() {
  document.querySelectorAll(".model-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const model = tab.dataset.model;
      document
        .querySelectorAll(".model-tab")
        .forEach((t) => t.classList.remove("active"));
      document
        .querySelectorAll(".model-pane")
        .forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document
        .querySelector(`.model-pane[data-model="${model}"]`)
        .classList.add("active");
    });
  });
}
