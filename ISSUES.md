# 🐛 AutoML Studio — Open Issues

This document tracks confirmed bugs open for contribution.
Each entry has a root-cause explanation and a suggested fix so you can jump straight in.

**Legend:** 🔴 High &nbsp; 🟡 Medium &nbsp; 🟢 Good first issue

---

## 🖥 UI Issues — Training Page

The training page (`templates/model_training.html`) is the most active area of the UI and has several layout and styling issues. Contributions here are especially welcome.

---

### UI-001

**🔴 Bootstrap CSS not imported — the 2-column training layout is broken**

**File:** `templates/base.html`, `templates/model_training.html`

The training page uses Bootstrap grid classes throughout — `container-fluid`, `row`, `col-lg-3`, `col-lg-9`, `col-md-4`, and Bootstrap spacing utilities like `px-4`, `mb-4`, `mt-3`. However, Bootstrap CSS is **never imported** in `base.html` or anywhere else.

Without Bootstrap, all of these classes have no effect. The intended layout — a narrow configuration panel on the left and a wide training area on the right — instead renders as two full-width stacked blocks on every screen size.

**Suggested fix:**

Add Bootstrap 5 CSS to `base.html`:

```html
<!-- In <head>, after the Font Awesome link -->
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
  crossorigin="anonymous">
```

Or — the preferred long-term approach — replace the Bootstrap grid markup in `model_training.html` entirely with a native CSS Grid or Flexbox layout, eliminating the Bootstrap dependency.

**Labels:** `bug` `ui` `good first issue`

---

### UI-002

**🔴 Local CSS in model_training.html conflicts with global styles**

**File:** `templates/model_training.html` (lines 771–828)

The training page redeclares several CSS classes already defined globally in `base.html`, with different values:

| Class | `base.html` | `model_training.html` override |
|---|---|---|
| `.btn` | `border-radius: 8px` | `border-radius: 4px` |
| `.btn-sec` | `background: var(--bg)` | `background: white` |
| `.stats-grid` | no column definition | `grid-template-columns: repeat(3, 1fr)` |

The local definitions win due to cascade order, so training page buttons look visually inconsistent with every other page in the app. The `.stats-grid` override also risks breaking the global stats layout if that class is ever reused.

**Suggested fix:**

1. Remove the `.btn` and `.btn-sec` re-declarations from `model_training.html` — the global versions from `base.html` are sufficient
2. Rename the training-specific `.stats-grid` override to a scoped class like `.training-stats-grid` so it doesn't bleed globally

**Labels:** `bug` `ui` `css`

---

### UI-003

**🟡 Progress bar hidden with `!important` — JavaScript updates silently fail**

**File:** `templates/model_training.html` (lines 1113–1117)

The progress bar elements are hidden in CSS:

```css
#progressFill,
#progressPercent,
#progressBarContainer {
  display: none !important;
}
```

But the JavaScript training-status polling function still calls `progressFill.style.width = percent + '%'` and attempts to show `progressBarContainer`. Because `display: none !important` cannot be overridden by inline styles, these JS updates silently do nothing — no error, no visible progress bar.

**Suggested fix:**

Either remove the `!important` rules and wire up the JS to correctly toggle visibility via `classList`, or remove both the dead CSS rules and the dead JS references if this progress bar has been fully replaced by the new one.

**Labels:** `bug` `ui` `javascript`

---

### UI-004

**🟡 Training page has no responsive handling between 1024px–1200px**

**File:** `templates/model_training.html`

The parameter cards grid collapses to a single column at `≤1200px`, but the outer `col-lg-3` / `col-lg-9` two-column split (once Bootstrap is fixed) activates at `≥992px`. In the 992–1100px range the left config panel is very narrow, causing hyperparameter controls and model selection cards to overflow or clip with no graceful fallback.

**Suggested fix:**

Add a media query at `max-width: 1100px` that either stacks the layout vertically or moves the config panel into a collapsible top section.

**Labels:** `enhancement` `ui` `responsive`

---

## 🪲 Other Issues

A few more things worth knowing. Some of these are intentionally left without full fix guidance — dig in and explore. 👀

---

### BUG-001

**🟡 No per-user session isolation on DATASTORE**

**File:** `state.py`

`DATASTORE` is a plain module-level Python dict shared across every request. Two users accessing the app simultaneously, or the same user opening two browser tabs, will overwrite each other's datasets and training state. This is a known architectural limitation of the current design.

**Labels:** `bug` `architecture`

---

### PERF-001

**🟢 Debug `print()` statements left throughout production code**

**Files:** `app.py`, `utility.py`

There are a large number of `print(f"DEBUG: ...")` and `print(f"  -> ...")` calls left in from development that clutter server output and make real log messages hard to spot. Replacing these with Python's `logging` module is a good, contained contribution.

```python
import logging
logger = logging.getLogger(__name__)

# replace: print(f"DEBUG: X shape {X.shape}")
# with:    logger.debug("X shape: %s", X.shape)
```

**Labels:** `performance` `code-quality` `good first issue`

---

## 🗺️ Feature Roadmap

Want to add something new? These are well-scoped additions the project would benefit from:

| Idea | Difficulty |
|---|---|
| Add XGBoost & LightGBM to `models_registry.py` | 🟡 Medium |
| Add SHAP explainability plots to the Results page | 🟡 Medium |
| Add Excel (`.xlsx`) upload support | 🟢 Easy |
| Add Dockerfile for containerised deployment | 🟡 Medium |
| Unit tests for `utility.py` core functions | 🟡 Medium |
| Dark mode toggle | 🟢 Easy |

---

*Found something not listed here? [Open an issue](https://github.com/Palakdeep-Singh/Auto-ML/issues/new) — the more detail the better.*
