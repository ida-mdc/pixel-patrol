# Installation

Answer the questions below and we'll show you the exact commands for your setup.

---

<div class="pp-wizard" id="pp-inst-wizard">

  <!-- Command preview -->
  <div class="wiz-preview">
    <div class="wiz-preview-header">
      <span class="wiz-preview-label">Your commands</span>
      <button class="wiz-copy-btn" id="inst-copy-btn" onclick="instWiz.copy()">Copy</button>
    </div>
    <pre class="wiz-code-pre" id="inst-cmd"># Answer the questions below to build your install commands</pre>
  </div>

  <!-- Q1: OS -->
  <div class="wiz-step wiz-visible" id="iws-os">
    <div class="wiz-step-q">What operating system are you on?</div>
    <div class="wiz-options wiz-options-row">
      <label class="wiz-option">
        <input type="radio" name="inst-os" value="unix"
               onchange="instWiz.pick('os','unix')">
        <div>
          <div class="wiz-opt-title">macOS or Linux</div>
        </div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="inst-os" value="windows"
               onchange="instWiz.pick('os','windows')">
        <div>
          <div class="wiz-opt-title">Windows</div>
        </div>
      </label>
    </div>
  </div>

  <!-- Q2: uv? (hidden until OS is chosen) -->
  <div class="wiz-step wiz-hidden" id="iws-uv">
    <div class="wiz-step-q">Do you have <code>uv</code> installed?</div>
    <div class="wiz-step-hint">
      <code>uv</code> is a fast Python package manager — we recommend it, but pip works too.
      Not sure? Run <code>uv --version</code> in your terminal.
    </div>
    <div class="wiz-options wiz-options-row">
      <label class="wiz-option">
        <input type="radio" name="inst-uv" value="yes"
               onchange="instWiz.pick('uv','yes')">
        <div><div class="wiz-opt-title">Yes, I have uv</div></div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="inst-uv" value="no"
               onchange="instWiz.pick('uv','no')">
        <div><div class="wiz-opt-title">No, install it for me</div></div>
      </label>
      <label class="wiz-option">
        <input type="radio" name="inst-uv" value="pip"
               onchange="instWiz.pick('uv','pip')">
        <div>
          <div class="wiz-opt-title">I'd rather use pip</div>
          <div class="wiz-opt-sub">⚠️ Installing without a virtual environment can conflict with other packages in your Python installation. We recommend using uv instead.</div>
        </div>
      </label>
    </div>
  </div>

  <!-- Done message -->
  <div class="wiz-done-msg" id="inst-done">
    ✓ Your install commands are ready above. Copy and run them in your terminal.
  </div>

</div>

<script>
const instWiz = {
  state: { os: null, uv: null },

  show(id) {
    const el = document.getElementById('iws-' + id);
    if (el && el.classList.contains('wiz-hidden')) {
      el.classList.remove('wiz-hidden');
      el.classList.add('wiz-visible');
    }
  },

  pick(field, value) {
    this.state[field] = value;
    document.querySelectorAll('input[name="inst-' + field + '"]').forEach(r => {
      r.closest('.wiz-option').classList.toggle('wiz-selected', r.value === value);
    });
    this.reveal();
    this.rebuild();
  },

  reveal() {
    const s = this.state;
    if (s.os) this.show('uv');
    if (s.os && s.uv) {
      const done = document.getElementById('inst-done');
      if (done) { done.classList.add('wiz-visible'); }
    }
  },

  rebuild() {
    const s = this.state;
    if (!s.os) {
      document.getElementById('inst-cmd').textContent =
        '# Answer the questions below to build your install commands';
      return;
    }

    const isWin   = s.os === 'windows';
    const activate = isWin
      ? '.venv\\Scripts\\Activate.ps1'
      : 'source .venv/bin/activate';

    let lines = [];

    if (!s.uv) {
      // OS chosen, waiting for uv answer
      document.getElementById('inst-cmd').textContent =
        '# Choose your preferred install method above';
      return;
    }

    if (s.uv === 'no') {
      // Install uv first
      if (isWin) {
        lines.push('# 1. Install uv');
        lines.push('powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"');
        lines.push('');
      } else {
        lines.push('# 1. Install uv');
        lines.push('curl -Ls https://astral.sh/uv/install.sh | sh');
        lines.push('# Restart your terminal, then continue:');
        lines.push('');
      }
    }

    if (s.uv === 'pip') {
      lines.push('# Install Pixel Patrol with pip');
      lines.push('pip install pixel-patrol');
      lines.push('');
      lines.push('# Verify');
      lines.push('pixel-patrol --version');
    } else {
      const step = s.uv === 'no' ? '2' : '1';
      lines.push('# ' + step + '. Create a virtual environment');
      lines.push('uv venv --python 3.12 .venv');
      lines.push('');
      lines.push('# ' + (parseInt(step)+1) + '. Activate it');
      lines.push(activate);
      lines.push('');
      lines.push('# ' + (parseInt(step)+2) + '. Install Pixel Patrol');
      lines.push('uv pip install pixel-patrol');
      lines.push('');
      lines.push('# ' + (parseInt(step)+3) + '. Verify');
      lines.push('pixel-patrol --version');
    }

    document.getElementById('inst-cmd').textContent = lines.join('\n');
  },

  copy() {
    const text = document.getElementById('inst-cmd').textContent;
    if (text.startsWith('#')) {
      // If it's just a placeholder comment, do nothing meaningful
    }
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('inst-copy-btn');
      const prev = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = prev; }, 2000);
    });
  }
};
</script>

---

!!! tip "Already installed?"
    Run `pixel-patrol --version` to confirm. Then head to the [Processing tutorial](processing.md) to build your first command.
