# 🤝 Contributing to AutoML Studio

Thanks for your interest in contributing! This guide covers everything from setting up your environment to getting your PR merged.

---

## Table of Contents

- [Getting Started](#-getting-started)
- [How to Contribute](#-how-to-contribute)
- [Development Setup](#-development-setup)
- [Coding Standards](#-coding-standards)
- [Commit Convention](#-commit-convention)
- [Pull Request Process](#-pull-request-process)
- [Good First Issues](#-good-first-issues)

---

## 🚀 Getting Started

### Prerequisites
- Python **3.9+**
- Git
- A GitHub account

### Fork & Clone

```bash
# 1. Fork the repo — click "Fork" top-right on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/Auto-ML.git
cd Auto-ML

# 3. Add upstream remote to stay in sync
git remote add upstream https://github.com/Palakdeep-Singh/Auto-ML.git
```

### Set Up Your Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
# Visit http://127.0.0.1:5000 — upload a sample CSV and walk the full pipeline
```

If everything runs end-to-end you're ready. ✅

---

## 💡 How to Contribute

### Found a bug?
1. Search [existing issues](https://github.com/Palakdeep-Singh/Auto-ML/issues) and check [`ISSUES.md`](./ISSUES.md) first
2. If it's new, open a **Bug Report** issue with: reproduction steps, expected vs actual behaviour, and screenshots for UI bugs

### Have a feature idea?
Open a **Feature Request** issue describing the problem, your proposed solution, and whether you'd like to build it.

### Want to pick up an existing issue?
Comment on the issue so others know you're working on it, then follow the [Submitting Code](#submitting-code) steps below.

### Submitting Code

```bash
# 1. Sync your fork first
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create a branch
git checkout -b fix/your-fix-name

# 3. Make changes, then commit
git add .
git commit -m "fix(ui): describe your change"

# 4. Push and open a Pull Request
git push origin fix/your-fix-name
```

---

## 🛠 Development Setup

### Key files at a glance

| File | Purpose |
|---|---|
| `app.py` | All Flask routes — start here for any endpoint change |
| `utility.py` | All ML logic — cleaning, training, feature engineering |
| `models_registry.py` | Add or edit models and their hyperparameter configs |
| `state.py` | Shared runtime state between requests |
| `templates/base.html` | Global layout, CSS variables, sidebar |
| `templates/model_training.html` | Training dashboard — main area with open UI issues |

### Adding a new model

Everything lives in `models_registry.py`. Add an entry like this:

```python
"xgboost_regressor": {
    "label": "XGBoost Regressor",
    "model": XGBRegressor,
    "params": {
        "model__n_estimators": {
            "type": "int", "min": 50, "max": 500, "step": 50, "default": 100
        },
        "model__learning_rate": {
            "type": "float", "min": 0.01, "max": 0.3, "step": 0.01, "default": 0.1
        },
    }
},
```

Then add it to `MODEL_COST` in `state.py` and add the package to `requirements.txt`.

---

## 📐 Coding Standards

**Python**
- Follow PEP 8
- Use type hints where practical
- Replace any `print()` debug statements you encounter with `logging`:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.debug("X shape: %s", X.shape)  # instead of print()
  ```
- No hardcoded paths or magic strings

**HTML / CSS**
- Page-specific styles belong in that template's `<style>` block
- Shared styles belong in `static/custom.css`
- Do **not** redefine class names that already exist in `base.html` — create a scoped name instead
- Test at desktop, tablet (1024px), and mobile (375px) before submitting

**JavaScript**
- Vanilla JS only — no new libraries without prior discussion
- Use `const` / `let`, not `var`
- Handle errors explicitly — don't swallow exceptions silently

---

## 📝 Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>
```

| Type | Use for |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructure, no behaviour change |
| `style` | Formatting, no logic change |
| `perf` | Performance improvement |
| `test` | Tests |
| `chore` | Tooling, config, CI |

**Examples:**
```
fix(ui): add Bootstrap 5 CSS import to base.html
fix(ui): remove duplicate .btn definitions from model_training.html
feat(models): add XGBoost and LightGBM to models_registry
perf: replace print() calls with logging in utility.py
docs: update README app flow section
```

---

## 🔀 Pull Request Process

**Branch naming:**
```
fix/bootstrap-css-import
feat/add-xgboost-support
docs/update-contributing-guide
refactor/state-session-isolation
```

**Before submitting, check:**
- [ ] App starts and runs without errors
- [ ] You've tested your change end-to-end in the browser
- [ ] No debug `print()` statements added
- [ ] Commit messages follow the convention above
- [ ] First PR? Add yourself to [`CONTRIBUTORS.md`](./CONTRIBUTORS.md)

**PR description should include:**
- What the PR does (1–2 sentences)
- `Closes #<issue-number>`
- Screenshots for any UI changes
- Any known caveats

A maintainer will review within a few days. Address comments with new commits — don't force-push unless asked.

---

## 🌱 Good First Issues

| Issue | File | Effort |
|---|---|---|
| [PERF-001](./ISSUES.md#perf-001) — Replace `print()` with `logging` | `app.py`, `utility.py` | ~1 hr |
| [UI-001](./ISSUES.md#ui-001) — Add Bootstrap CSS import | `base.html` | ~20 min |
| [UI-002](./ISSUES.md#ui-002) — Fix CSS class conflicts on training page | `model_training.html` | ~30 min |
| [UI-003](./ISSUES.md#ui-003) — Fix hidden progress bar | `model_training.html` | ~20 min |
| Add Excel upload support | `app.py`, `upload.html` | ~1 hr |
| Add dark mode toggle | `base.html`, `custom.css` | ~1 hr |

---

## 📬 Questions?

Open a [Discussion](https://github.com/Palakdeep-Singh/Auto-ML/discussions) or email **palakdeep8055@gmail.com**.

*Every contribution, no matter how small, is appreciated. 🙏*
