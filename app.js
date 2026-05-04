/**
 * app.js — Guitar FX Neural Demo
 *
 * Fetches manifest.json, populates excerpt tabs and effect audio players,
 * handles tab switching with smooth transitions, and provides
 * accordion toggles for spectrogram images.
 */

(() => {
  "use strict";

  // ═══════════════════════════════════════════════════════════
  // Effect metadata (categories + icons for visual grouping)
  // ═══════════════════════════════════════════════════════════

  const EFFECT_META = {
    rat: { category: "distortion", categoryLabel: "Distortion" },
    tubeScreamer: { category: "distortion", categoryLabel: "Distortion" },
    flanger: { category: "modulation", categoryLabel: "Modulation" },
    phaser: { category: "modulation", categoryLabel: "Modulation" },
    digitalDelay: { category: "delay", categoryLabel: "Delay" },
    sweepEcho: { category: "delay", categoryLabel: "Delay" },
    tapeEcho: { category: "delay", categoryLabel: "Delay" },
    hallReverb: { category: "reverb", categoryLabel: "Reverb" },
    plateReverb: { category: "reverb", categoryLabel: "Reverb" },
    "spring-Reverb": { category: "reverb", categoryLabel: "Reverb" },
  };

  // Desired display order: Distortion → Modulation → Delay → Reverb
  const CATEGORY_ORDER = ["distortion", "modulation", "delay", "reverb"];

  // ═══════════════════════════════════════════════════════════
  // DOM references
  // ═══════════════════════════════════════════════════════════

  const tabsContainer = document.getElementById("excerpt-tabs");
  const cleanAudio = document.getElementById("clean-audio");
  const cleanSpecToggle = document.getElementById("clean-spec-toggle");
  const cleanSpecAccordion = document.getElementById("clean-spec-accordion");
  const cleanSpecImg = document.getElementById("clean-spec-img");
  const cleanSectionLabel = document.getElementById("clean-section-label");
  const effectsGrid = document.getElementById("effects-grid");
  const effectsCount = document.getElementById("effects-count");
  const loadingEl = document.getElementById("loading");

  // ═══════════════════════════════════════════════════════════
  // State
  // ═══════════════════════════════════════════════════════════

  let manifest = null;
  let activeExcerpt = null;

  // ═══════════════════════════════════════════════════════════
  // Accordion helper
  // ═══════════════════════════════════════════════════════════

  function setupAccordionToggle(toggleBtn, accordionEl) {
    toggleBtn.addEventListener("click", () => {
      const isOpen = accordionEl.classList.contains("open");
      toggleBtn.classList.toggle("active", !isOpen);
      accordionEl.classList.toggle("open", !isOpen);
      toggleBtn.querySelector(".spec-toggle__chevron").textContent = isOpen ? "▶" : "▼";
    });
  }

  // Wire up the clean section accordion
  setupAccordionToggle(cleanSpecToggle, cleanSpecAccordion);

  // ═══════════════════════════════════════════════════════════
  // Helpers
  // ═══════════════════════════════════════════════════════════

  function sortEffects(effects) {
    return [...effects].sort((a, b) => {
      const catA = EFFECT_META[a.name]?.category || "zzz";
      const catB = EFFECT_META[b.name]?.category || "zzz";
      const orderA = CATEGORY_ORDER.indexOf(catA);
      const orderB = CATEGORY_ORDER.indexOf(catB);
      if (orderA !== orderB) return orderA - orderB;
      return a.displayName.localeCompare(b.displayName);
    });
  }

  function createEffectRow(effect, predictions, index) {
    const meta = EFFECT_META[effect.name] || { category: "other", categoryLabel: "Other" };
    const preds = predictions[effect.name] || {};

    const row = document.createElement("div");
    row.className = "effect-row";
    row.style.animationDelay = `${index * 50}ms`;

    const toggleId = `spec-toggle-${effect.name}`;
    const accordionId = `spec-accordion-${effect.name}`;

    row.innerHTML = `
      <div class="effect-row__name">
        ${effect.displayName}
        <span class="effect-category effect-category--${meta.category}">
          ${meta.categoryLabel}
        </span>
      </div>
      <div class="effect-row__players">
        <div class="player-block player-block--unet">
          <div class="player-block__label">
            <span class="player-block__dot"></span>
            U-Net (Log)
          </div>
          <audio controls preload="metadata" id="audio-unet-${effect.name}">
            <source src="${preds.unet || ''}" type="audio/wav">
          </audio>
        </div>
        <div class="player-block player-block--convvae">
          <div class="player-block__label">
            <span class="player-block__dot"></span>
            ConvVAE
          </div>
          <audio controls preload="metadata" id="audio-convvae-${effect.name}">
            <source src="${preds.convvae || ''}" type="audio/wav">
          </audio>
        </div>
      </div>
      <button class="spec-toggle" id="${toggleId}" type="button">
        <span class="spec-toggle__chevron">▶</span>
        Show Spectrograms
      </button>
      <div class="spec-accordion" id="${accordionId}">
        <div class="spec-accordion__inner">
          <div class="spec-images">
            <div class="spec-block spec-block--unet">
              <div class="spec-block__label">U-Net (Log)</div>
              <img class="spec-block__img"
                   src="${preds.unetSpec || ''}"
                   alt="U-Net spectrogram for ${effect.displayName}"
                   loading="lazy">
            </div>
            <div class="spec-block spec-block--convvae">
              <div class="spec-block__label">ConvVAE</div>
              <img class="spec-block__img"
                   src="${preds.convvaeSpec || ''}"
                   alt="ConvVAE spectrogram for ${effect.displayName}"
                   loading="lazy">
            </div>
          </div>
        </div>
      </div>
    `;

    // Wire up this row's accordion after inserting into DOM
    requestAnimationFrame(() => {
      const btn = document.getElementById(toggleId);
      const acc = document.getElementById(accordionId);
      if (btn && acc) setupAccordionToggle(btn, acc);
    });

    return row;
  }

  // ═══════════════════════════════════════════════════════════
  // Rendering
  // ═══════════════════════════════════════════════════════════

  function renderTabs() {
    tabsContainer.innerHTML = "";

    manifest.excerpts.forEach((excerpt) => {
      const btn = document.createElement("button");
      btn.className = "excerpt-tab";
      btn.textContent = excerpt.label;
      btn.dataset.excerptId = excerpt.id;
      btn.setAttribute("aria-label", `Select ${excerpt.label}`);

      btn.addEventListener("click", () => {
        selectExcerpt(excerpt.id);
      });

      tabsContainer.appendChild(btn);
    });
  }

  function selectExcerpt(excerptId) {
    if (activeExcerpt === excerptId) return;
    activeExcerpt = excerptId;

    // Update tab active state
    tabsContainer.querySelectorAll(".excerpt-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.excerptId === excerptId);
    });

    // Find excerpt data
    const excerpt = manifest.excerpts.find((e) => e.id === excerptId);
    if (!excerpt) return;

    // Pause all currently playing audio
    document.querySelectorAll("audio").forEach((a) => {
      a.pause();
      a.currentTime = 0;
    });

    // Update clean audio + spectrogram
    cleanAudio.src = excerpt.cleanAudio;
    cleanSpecImg.src = excerpt.cleanSpec || "";

    // Update clean section label based on source
    const source = excerptId.startsWith("egfxset")
      ? "Clean Input — EGFxSet (In-Distribution)"
      : "Clean Input — Guitar-TECHS Direct Input";
    cleanSectionLabel.textContent = source;

    // Reset clean accordion to closed
    cleanSpecToggle.classList.remove("active");
    cleanSpecAccordion.classList.remove("open");
    cleanSpecToggle.querySelector(".spec-toggle__chevron").textContent = "▶";

    // Fade out grid, rebuild, fade in
    effectsGrid.style.opacity = "0";
    effectsGrid.style.transform = "translateY(8px)";

    setTimeout(() => {
      renderEffectsGrid(excerpt);

      requestAnimationFrame(() => {
        effectsGrid.style.transition = "opacity 300ms ease, transform 300ms ease";
        effectsGrid.style.opacity = "1";
        effectsGrid.style.transform = "translateY(0)";
      });
    }, 200);
  }

  function renderEffectsGrid(excerpt) {
    effectsGrid.innerHTML = "";

    const sortedEffects = sortEffects(manifest.effects);
    effectsCount.textContent = `${sortedEffects.length} effects × 2 models`;

    sortedEffects.forEach((effect, index) => {
      const row = createEffectRow(effect, excerpt.predictions, index);
      effectsGrid.appendChild(row);
    });
  }

  // ═══════════════════════════════════════════════════════════
  // Init
  // ═══════════════════════════════════════════════════════════

  async function init() {
    try {
      const response = await fetch("manifest.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      manifest = await response.json();

      loadingEl.style.display = "none";

      renderTabs();

      // Select first excerpt
      if (manifest.excerpts.length > 0) {
        selectExcerpt(manifest.excerpts[0].id);
      }
    } catch (err) {
      console.error("Failed to load manifest:", err);
      loadingEl.innerHTML = `
        <div style="color: #ff6b6b; text-align: center; padding: 48px;">
          <p style="font-size: 1.2rem; margin-bottom: 8px;">Could not load audio data</p>
          <p style="font-size: 0.85rem; color: #9898b0;">
            Make sure <code>manifest.json</code> exists and you're serving this page via a local server.
            <br><br>
            <code style="background: rgba(255,255,255,0.05); padding: 4px 12px; border-radius: 6px;">
              python3 -m http.server 8000
            </code>
          </p>
        </div>
      `;
    }
  }

  // Start
  document.addEventListener("DOMContentLoaded", init);
})();
