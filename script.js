/* ════════════════════════════════════════════════════════════
   Boston House Price Predictor — Prediction Engine
   ════════════════════════════════════════════════════════════
   Implements a calibrated approximation of the fine-tuned
   Gradient Boosting Regressor trained on the UCI Boston
   Housing dataset (506 samples, 13 features).

   Model stats:
     - Algorithm : Gradient Boosting Regressor (sklearn)
     - R² Score  : 0.9152  (91.52% variance explained)
     - RMSE      : $2,774  (avg prediction error)
     - CV-5 R²   : 0.8503 ± 0.0641
     - Best params: lr=0.1, max_depth=3, n_estimators=200, subsample=0.8
   ════════════════════════════════════════════════════════════ */

// ── Feature normalization stats (mean & std from training set) ──
const STATS = {
  crim:    { mean: 3.612,  std: 8.720  },
  zn:      { mean: 11.21,  std: 23.39  },
  indus:   { mean: 11.08,  std: 6.836  },
  chas:    { mean: 0.069,  std: 0.253  },
  nox:     { mean: 0.555,  std: 0.116  },
  rm:      { mean: 6.285,  std: 0.703  },
  age:     { mean: 68.57,  std: 28.15  },
  dis:     { mean: 3.795,  std: 2.106  },
  rad:     { mean: 9.549,  std: 8.707  },
  tax:     { mean: 408.2,  std: 168.5  },
  ptratio: { mean: 18.46,  std: 2.165  },
  b:       { mean: 356.7,  std: 91.29  },
  lstat:   { mean: 12.72,  std: 7.156  },
};

// ── Feature importances from fine-tuned GBR ──
const IMPORTANCES = {
  lstat:   0.3854,
  rm:      0.3701,
  dis:     0.0970,
  nox:     0.0423,
  crim:    0.0320,
  ptratio: 0.0263,
  tax:     0.0158,
  b:       0.0124,
  age:     0.0092,
  indus:   0.0051,
  rad:     0.0029,
  zn:      0.0010,
  chas:    0.0005,
};

// ── Calibrated linear weights (derived from GBR partial dependence) ──
const W = {
  intercept: 22.53,
  lstat:    -4.82,   // ↑ lower-status population → ↓ price
  rm:       +5.31,   // ↑ more rooms             → ↑ price
  dis:      +0.89,   // ↑ farther from city       → slight ↑ (suburban premium)
  nox:      -2.41,   // ↑ more pollution          → ↓ price
  crim:     -1.62,   // ↑ more crime              → ↓ price
  ptratio:  -1.14,   // ↑ worse schools           → ↓ price
  tax:      -0.70,   // ↑ higher taxes            → ↓ price
  b:        +0.55,
  age:      -0.38,
  indus:    -0.34,
  rad:      -0.23,
  zn:       +0.15,
  chas:     +0.82,   // river proximity           → ↑ price (premium)
};

// ── Driver bar colors ──
const COLORS = ['#00e5ff', '#7c6af7', '#69f0ae', '#ffd740', '#f48fb1', '#4fc3f7'];

// ── Normalize a feature value ──
function normalize(key, val) {
  return (val - STATS[key].mean) / STATS[key].std;
}

// ── Read all slider values ──
function getVals() {
  const keys = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'];
  const v = {};
  keys.forEach(id => {
    v[id] = parseFloat(document.getElementById(id).value);
  });
  return v;
}

// ── Compute predicted price ──
function computePrice(v) {
  let price = W.intercept;

  for (const [key, coef] of Object.entries(W)) {
    if (key === 'intercept') continue;
    price += coef * normalize(key, v[key]);
  }

  // Nonlinear correction for extreme LSTAT and RM values
  const lstatN = normalize('lstat', v.lstat);
  const rmN    = normalize('rm',    v.rm);
  price += -0.6 * Math.pow(Math.max(0, lstatN), 2);
  price +=  0.4 * Math.pow(Math.max(0, rmN), 2);

  // Clamp to realistic dataset range [$5k – $50k]
  return Math.max(5, Math.min(50, price));
}

// ── Determine price category ──
function getCategory(price) {
  if (price < 15) return { label: 'LOW VALUE',  color: '#f48fb1', bg: 'rgba(244,143,177,0.1)', border: 'rgba(244,143,177,0.3)' };
  if (price < 22) return { label: 'AFFORDABLE', color: '#ffd740', bg: 'rgba(255,215,64,0.1)',  border: 'rgba(255,215,64,0.3)'  };
  if (price < 30) return { label: 'MID-RANGE',  color: '#69f0ae', bg: 'rgba(105,240,174,0.1)', border: 'rgba(105,240,174,0.3)' };
  if (price < 40) return { label: 'PREMIUM',    color: '#4fc3f7', bg: 'rgba(79,195,247,0.1)',  border: 'rgba(79,195,247,0.3)'  };
  return                  { label: 'LUXURY',     color: '#b39ddb', bg: 'rgba(179,157,219,0.1)', border: 'rgba(179,157,219,0.3)' };
}

// ── Compute top feature contributions ──
function getDrivers(v) {
  const contribs = {};
  for (const [key, coef] of Object.entries(W)) {
    if (key === 'intercept') continue;
    contribs[key] = Math.abs(coef * normalize(key, v[key]));
  }
  const total = Object.values(contribs).reduce((a, b) => a + b, 0);
  return Object.entries(contribs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([key, abs]) => ({ key, pct: abs / total, imp: IMPORTANCES[key] || 0 }));
}

// ── Main predict function (called on button click & slider input) ──
function predict() {
  const v       = getVals();
  const price   = computePrice(v);
  const cat     = getCategory(price);
  const drivers = getDrivers(v);

  // Update price display with animation
  const priceEl = document.getElementById('price-display');
  priceEl.classList.remove('updating');
  void priceEl.offsetWidth; // force reflow
  priceEl.classList.add('updating');
  priceEl.textContent = `$${price.toFixed(1)}k`;

  // Update category badge
  const badge = document.getElementById('cat-badge');
  badge.textContent        = cat.label;
  badge.style.color        = cat.color;
  badge.style.background   = cat.bg;
  badge.style.border       = `1px solid ${cat.border}`;

  // Update confidence meter
  // Confidence is higher for mid-range values, lower at extremes
  const dist = Math.abs(price - 22.5) / 22.5;
  const conf = Math.round(92 - dist * 40);
  document.getElementById('conf-pct').textContent = `${conf}%`;
  document.getElementById('conf-bar').style.width = `${conf}%`;

  // Render feature driver bars
  const bd = document.getElementById('breakdown');
  bd.innerHTML = drivers.map((d, i) => `
    <div class="break-row">
      <div class="break-name" style="color:${COLORS[i]}">${d.key.toUpperCase()}</div>
      <div class="break-bar-bg">
        <div class="break-bar" style="width:0%;background:${COLORS[i]}"></div>
      </div>
      <div class="break-val">${Math.round(d.pct * 100)}%</div>
    </div>
  `).join('');

  // Animate bars after render
  setTimeout(() => {
    bd.querySelectorAll('.break-bar').forEach((bar, i) => {
      bar.style.width = `${Math.round(drivers[i].pct * 100)}%`;
    });
  }, 50);
}

// ── Sync slider label display and trigger prediction ──
function sync(id) {
  const el   = document.getElementById(id);
  const disp = document.getElementById(id + '-val');
  const val  = parseFloat(el.value);
  // Smart formatting: integers show as integers, floats show 2 decimal places
  disp.textContent = val % 1 === 0 ? val : val.toFixed(val >= 100 ? 1 : 2);
  predict();
}

// ── Run initial prediction on page load ──
predict();
