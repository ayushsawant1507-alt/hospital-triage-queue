/**
 * Hospital / Emergency Triage Queue — Interactive Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Navigation tabs
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.target);
      if (target) target.classList.add('active');
    });
  });

  // Theme Toggle (Light / Dark)
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const themeIcon = document.getElementById('theme-icon');
  const themeLabel = document.getElementById('theme-label');

  function applyTheme(theme) {
    if (theme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
      if (themeIcon) themeIcon.textContent = '🌙';
      if (themeLabel) themeLabel.textContent = 'Dark Mode';
    } else {
      document.documentElement.removeAttribute('data-theme');
      if (themeIcon) themeIcon.textContent = '☀️';
      if (themeLabel) themeLabel.textContent = 'Light Mode';
    }
  }

  const savedTheme = localStorage.getItem('triage_theme') || 'dark';
  applyTheme(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const nextTheme = isLight ? 'dark' : 'light';
      localStorage.setItem('triage_theme', nextTheme);
      applyTheme(nextTheme);
      fetchQueueState();
    });
  }

  // Severity color maps
  const sevColors = {
    1: '#ff3b5c', // Critical
    2: '#ff8c2b', // Emergent
    3: '#ffd13b', // Urgent
    4: '#00d2d3', // Less Urgent
    5: '#8395a7', // Non-Urgent
  };

  // -------------------------------------------------------------------------
  // TAB 1: Live Triage & Min-Heap Visualizer
  // -------------------------------------------------------------------------
  const heapSvg = document.getElementById('heap-svg');
  const heapArrayView = document.getElementById('heap-array-view');
  const queueTableBody = document.getElementById('queue-table-body');
  const queueCountEl = document.getElementById('queue-count');
  const compCountEl = document.getElementById('global-comps');
  const moveCountEl = document.getElementById('global-moves');
  const nextPatientCard = document.getElementById('next-patient-card');

  const admitForm = document.getElementById('admit-form');
  const treatBtn = document.getElementById('treat-btn');
  const resetBtn = document.getElementById('reset-btn');
  const deteriorateSelect = document.getElementById('deteriorate-select');
  const deteriorateBtn = document.getElementById('deteriorate-btn');
  const quickCriticalBtn = document.getElementById('quick-critical-btn');

  async function fetchQueueState() {
    try {
      const res = await fetch('/api/queue/state');
      const data = await res.json();
      renderQueue(data);
    } catch (err) {
      console.error('Failed to fetch queue state', err);
    }
  }

  function renderQueue(state) {
    queueCountEl.textContent = state.length;
    compCountEl.textContent = state.comparisons;
    moveCountEl.textContent = state.moves;

    // Render Next Patient Card
    if (state.peek) {
      nextPatientCard.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <span class="badge badge-sev-${state.peek.severity}">${state.peek.severity_name}</span>
            <h3 style="margin-top:0.4rem; font-size:1.15rem; font-weight:700;">${state.peek.name}</h3>
            <p style="font-size:0.75rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">ID: #${state.peek.id} • Arrival #${state.peek.arrival_seq}</p>
          </div>
          <button id="card-treat-btn" class="btn btn-danger btn-sm">Treat Now</button>
        </div>
      `;
      document.getElementById('card-treat-btn')?.addEventListener('click', handleTreat);
    } else {
      nextPatientCard.innerHTML = `<p style="color:var(--text-muted); font-size:0.85rem;">Queue is empty. No waiting patients.</p>`;
    }

    // Render Table
    queueTableBody.innerHTML = '';
    deteriorateSelect.innerHTML = '<option value="">-- Select Patient --</option>';

    state.heap.forEach((p, idx) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace; color:var(--text-dim);">${idx}</td>
        <td style="font-weight:600;">#${p.id}</td>
        <td>${p.name}</td>
        <td><span class="badge badge-sev-${p.severity}">${p.severity_name}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;">${p.arrival_seq}</td>
      `;
      queueTableBody.appendChild(row);

      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `#${p.id} - ${p.name} (${p.severity_name})`;
      deteriorateSelect.appendChild(opt);
    });

    // Render Heap Array View
    heapArrayView.innerHTML = '';
    state.heap.forEach((p, idx) => {
      const cell = document.createElement('div');
      cell.className = `array-cell ${idx === 0 ? 'is-root' : ''}`;
      cell.innerHTML = `
        <div class="idx">[${idx}]</div>
        <div class="p-name">${p.name.split(' ')[0]}</div>
        <span class="badge badge-sev-${p.severity}" style="font-size:0.65rem; padding:0.1rem 0.4rem;">${p.severity_name}</span>
      `;
      heapArrayView.appendChild(cell);
    });

    // Render SVG Binary Tree
    renderBinaryTree(state.heap);
  }

  function renderBinaryTree(heap) {
    heapSvg.innerHTML = '';
    if (!heap || heap.length === 0) {
      heapSvg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#64748b" font-size="14">Heap is empty</text>`;
      return;
    }

    const width = heapSvg.clientWidth || 700;
    const height = 360;
    const nodeRadius = 22;

    // Calculate node positions
    const positions = [];
    const n = heap.length;
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const edgeColor = isLight ? 'rgba(0, 0, 0, 0.22)' : 'rgba(255, 255, 255, 0.15)';
    const nodeFill = isLight ? '#ffffff' : '#101522';
    const nameColor = isLight ? '#0f172a' : '#cbd5e1';

    for (let i = 0; i < n; i++) {
      const level = Math.floor(Math.log2(i + 1));
      const posInLevel = i - (Math.pow(2, level) - 1);
      const totalInLevel = Math.pow(2, level);

      const y = 50 + level * 75;
      const x = (width / (totalInLevel + 1)) * (posInLevel + 1);
      positions.push({ x, y });
    }

    // 1. Draw Connecting Edges
    for (let i = 0; i < n; i++) {
      const left = 2 * i + 1;
      const right = 2 * i + 2;

      if (left < n) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', positions[i].x);
        line.setAttribute('y1', positions[i].y);
        line.setAttribute('x2', positions[left].x);
        line.setAttribute('y2', positions[left].y);
        line.setAttribute('stroke', edgeColor);
        line.setAttribute('stroke-width', '2');
        heapSvg.appendChild(line);
      }

      if (right < n) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', positions[i].x);
        line.setAttribute('y1', positions[i].y);
        line.setAttribute('x2', positions[right].x);
        line.setAttribute('y2', positions[right].y);
        line.setAttribute('stroke', edgeColor);
        line.setAttribute('stroke-width', '2');
        heapSvg.appendChild(line);
      }
    }

    // 2. Draw Nodes
    for (let i = 0; i < n; i++) {
      const p = heap[i];
      const pos = positions[i];
      const color = sevColors[p.severity] || '#38bdf8';

      // Group
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.style.cursor = 'pointer';

      // Glow circle
      const glow = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      glow.setAttribute('cx', pos.x);
      glow.setAttribute('cy', pos.y);
      glow.setAttribute('r', nodeRadius + 3);
      glow.setAttribute('fill', color);
      glow.setAttribute('opacity', isLight ? '0.12' : '0.2');
      g.appendChild(glow);

      // Node circle
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pos.x);
      circle.setAttribute('cy', pos.y);
      circle.setAttribute('r', nodeRadius);
      circle.setAttribute('fill', nodeFill);
      circle.setAttribute('stroke', color);
      circle.setAttribute('stroke-width', i === 0 ? '3' : '2');
      g.appendChild(circle);

      // Label (Severity number)
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', pos.x);
      text.setAttribute('y', pos.y + 5);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', color);
      text.setAttribute('font-weight', 'bold');
      text.setAttribute('font-size', '13');
      text.setAttribute('font-family', 'JetBrains Mono');
      text.textContent = `S${p.severity}`;
      g.appendChild(text);

      // Name tooltip / bottom label
      const nameText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      nameText.setAttribute('x', pos.x);
      nameText.setAttribute('y', pos.y + 36);
      nameText.setAttribute('text-anchor', 'middle');
      nameText.setAttribute('fill', nameColor);
      nameText.setAttribute('font-size', '10');
      nameText.setAttribute('font-weight', '600');
      nameText.textContent = p.name.split(' ')[0];
      g.appendChild(nameText);

      heapSvg.appendChild(g);
    }
  }

  // Event Handlers for Queue
  admitForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('patient-name').value;
    const severity = document.getElementById('patient-severity').value;

    const res = await fetch('/api/queue/admit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, severity }),
    });
    const data = await res.json();
    if (data.success) {
      document.getElementById('patient-name').value = '';
      renderQueue(data.state);
    }
  });

  async function handleTreat() {
    try {
      const res = await fetch('/api/queue/extract', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        renderQueue(data.state);
      }
    } catch (err) {
      console.error(err);
    }
  }

  treatBtn.addEventListener('click', handleTreat);

  resetBtn.addEventListener('click', async () => {
    const res = await fetch('/api/queue/reset', { method: 'POST' });
    const data = await res.json();
    renderQueue(data.state);
  });

  deteriorateBtn.addEventListener('click', async () => {
    const id = deteriorateSelect.value;
    const severity = document.getElementById('new-severity-select').value;
    if (!id) return alert('Please select a patient first');

    const res = await fetch('/api/queue/update_severity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, severity }),
    });
    const data = await res.json();
    if (data.success) {
      renderQueue(data.state);
    }
  });

  quickCriticalBtn.addEventListener('click', async () => {
    const names = ['Logan Roy', 'Walter White', 'Tony Soprano', 'Thomas Shelby'];
    const randomName = names[Math.floor(Math.random() * names.length)];
    await fetch('/api/queue/admit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: randomName, severity: 1 }),
    });
    fetchQueueState();
  });

  // Initial load
  fetchQueueState();

  // -------------------------------------------------------------------------
  // TAB 2: Algorithm Sorting Arena
  // -------------------------------------------------------------------------
  const runSortsBtn = document.getElementById('run-sorts-btn');
  const sortResultsContainer = document.getElementById('sort-results-container');
  const sortSizeInput = document.getElementById('sort-size');
  const sortSizeVal = document.getElementById('sort-size-val');

  sortSizeInput.addEventListener('input', () => {
    sortSizeVal.textContent = sortSizeInput.value;
  });

  runSortsBtn.addEventListener('click', async () => {
    const n = sortSizeInput.value;
    const generator = document.getElementById('sort-generator').value;

    runSortsBtn.disabled = true;
    runSortsBtn.textContent = 'Sorting...';
    sortResultsContainer.innerHTML = '<p style="color:var(--text-muted);">Executing comparisons & moves across all 5 sorters...</p>';

    try {
      const res = await fetch('/api/sort/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n, generator }),
      });
      const data = await res.json();

      sortResultsContainer.innerHTML = '';
      data.results.forEach(r => {
        const card = document.createElement('div');
        card.className = 'sort-result-card';
        card.innerHTML = `
          <div>
            <div class="sort-name">${r.name}</div>
            <div class="sort-metrics">
              <span>Comparisons: <b>${r.comparisons.toLocaleString()}</b></span>
              <span>Moves: <b>${r.moves.toLocaleString()}</b></span>
              <span>Time: <b>${r.time_ms} ms</b></span>
            </div>
          </div>
          <div>
            <span class="badge ${r.is_sorted ? 'badge-sev-4' : 'badge-sev-1'}">
              ${r.is_sorted ? '✓ Correct' : '✗ Failed'}
            </span>
          </div>
        `;
        sortResultsContainer.appendChild(card);
      });
    } catch (err) {
      console.error(err);
    } finally {
      runSortsBtn.disabled = false;
      runSortsBtn.textContent = '⚡ Run Benchmark Race';
    }
  });

  // -------------------------------------------------------------------------
  // TAB 3: Search & Break-Even Visualizer
  // -------------------------------------------------------------------------
  const runSearchBtn = document.getElementById('run-search-btn');
  const searchResultsDiv = document.getElementById('search-results');
  const searchNInput = document.getElementById('search-n');

  runSearchBtn.addEventListener('click', async () => {
    const n = searchNInput.value;
    const target_id = document.getElementById('search-target-id').value || null;

    try {
      const res = await fetch('/api/search/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n, target_id }),
      });
      const data = await res.json();

      searchResultsDiv.innerHTML = `
        <div class="grid-2" style="margin-top:1rem;">
          <div class="glass-card">
            <h4 style="font-size:0.9rem; color:var(--text-muted); margin-bottom:0.5rem;">Sequential Linear Search</h4>
            <div style="font-size:1.8rem; font-weight:800; color:#ef4444; font-family:'JetBrains Mono';">${data.linear_comps}</div>
            <p style="font-size:0.75rem; color:var(--text-dim); margin-top:0.25rem;">Comparisons executed (O(n) scan)</p>
          </div>
          <div class="glass-card" style="border-color:var(--primary);">
            <h4 style="font-size:0.9rem; color:var(--text-muted); margin-bottom:0.5rem;">Binary Search (Iterative)</h4>
            <div style="font-size:1.8rem; font-weight:800; color:var(--primary); font-family:'JetBrains Mono';">${data.binary_comps}</div>
            <p style="font-size:0.75rem; color:var(--text-dim); margin-top:0.25rem;">Comparisons executed (O(log n) deferred check)</p>
          </div>
        </div>

        <div class="glass-card" style="margin-top:1rem; border-color:var(--border-accent);">
          <h4 style="font-size:0.95rem; font-weight:700; color:var(--text-main); margin-bottom:0.5rem;">
            Break-Even Proof (Master Theorem & Preprocessing Tradeoff)
          </h4>
          <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.5;">
            For a dataset of <b>n = ${data.n}</b> records, sorting takes ~n log₂ n comparisons.
            Each binary lookup takes ~log₂ n vs. n/2 for linear search.
            <br>
            <b>Break-even point:</b> Sorting pays for itself after exactly <b>${data.break_even_queries} lookup queries</b>!
          </p>
        </div>
      `;
    } catch (err) {
      console.error(err);
    }
  });

  // -------------------------------------------------------------------------
  // TAB 4: Discrete-Event ER Simulation
  // -------------------------------------------------------------------------
  const runSimBtn = document.getElementById('run-sim-btn');
  const simResultsDiv = document.getElementById('sim-results');

  runSimBtn.addEventListener('click', async () => {
    runSimBtn.disabled = true;
    runSimBtn.textContent = 'Simulating ER Operations...';

    const duration = document.getElementById('sim-duration').value;
    const doctors = document.getElementById('sim-doctors').value;
    const arrival_rate = document.getElementById('sim-rate').value;
    const aging_threshold = document.getElementById('sim-aging').value;

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration, doctors, arrival_rate, aging_threshold }),
      });
      const data = await res.json();

      simResultsDiv.innerHTML = `
        <div class="grid-2" style="margin-top:1.25rem;">
          <div class="glass-card" style="border-color:var(--primary);">
            <div class="card-header">
              <span class="card-title">Real-Time PriorityQueue (Min-Heap)</span>
              <span class="badge badge-sev-4">${data.priority_queue.throughput_hr} pts/hr</span>
            </div>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.75rem;">
              Total Treated: <b>${data.priority_queue.treated}</b> / ${data.priority_queue.admitted} admitted
            </p>
            <div style="font-size:0.82rem;">
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>Critical (Level 1) Wait:</span>
                <b style="color:var(--sev-1);">${data.priority_queue.avg_wait[1]} min (Max: ${data.priority_queue.max_wait[1]}m)</b>
              </div>
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>Emergent (Level 2) Wait:</span>
                <b style="color:var(--sev-2);">${data.priority_queue.avg_wait[2]} min</b>
              </div>
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>Urgent (Level 3) Wait:</span>
                <b style="color:var(--sev-3);">${data.priority_queue.avg_wait[3]} min</b>
              </div>
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0;">
                <span>Less Urgent (Level 4) Wait:</span>
                <b style="color:var(--sev-4);">${data.priority_queue.avg_wait[4]} min</b>
              </div>
            </div>
          </div>

          <div class="glass-card">
            <div class="card-header">
              <span class="card-title">Periodic Batch-Sort (Timsort)</span>
              <span class="badge badge-sev-2">${data.batch_sort.throughput_hr} pts/hr</span>
            </div>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.75rem;">
              Total Treated: <b>${data.batch_sort.treated}</b> / ${data.batch_sort.admitted} admitted
            </p>
            <div style="font-size:0.82rem;">
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>Critical (Level 1) Wait:</span>
                <b style="color:var(--sev-1);">${data.batch_sort.avg_wait[1]} min (Max: ${data.batch_sort.max_wait[1]}m)</b>
              </div>
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>Emergent (Level 2) Wait:</span>
                <b style="color:var(--sev-2);">${data.batch_sort.avg_wait[2]} min</b>
              </div>
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>Urgent (Level 3) Wait:</span>
                <b style="color:var(--sev-3);">${data.batch_sort.avg_wait[3]} min</b>
              </div>
              <div style="display:flex; justify-content:space-between; padding:0.3rem 0;">
                <span>Less Urgent (Level 4) Wait:</span>
                <b style="color:var(--sev-4);">${data.batch_sort.avg_wait[4]} min</b>
              </div>
            </div>
          </div>
        </div>
      `;
    } catch (err) {
      console.error(err);
    } finally {
      runSimBtn.disabled = false;
      runSimBtn.textContent = '🚀 Run Discrete-Event Simulation';
    }
  });
});
