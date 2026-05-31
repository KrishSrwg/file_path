# Learning Log

## Phase 0 — Structural Patterns & Edge Cases Worth Handling

### Document structures observed across 21 PDFs

1. **Single-brand with hybrid universal+specific structure** (most common). The 
   document is brand-specific but contains a small universal subsection (TB, 
   concomitant biologic restrictions, auth duration).

2. **Biosimilar umbrella structure**. Clinical criteria written once, applied 
   to all biosimilars; only quantity limits differ per biosimilar.

3. **Multi-biologic formulary with shared evaluation modules**. One document 
   covers many biologics using shared "Initial Evaluation" / "Renewal 
   Evaluation" sections; brand exceptions are embedded within them.

4. **Step Table grid structure**. Step therapy expressed as a multi-tier grid 
   (Step 1a, 1b, 2, 3a) specifying how many agents from each prior tier must 
   be failed.

5. **Brand-specific with no shared sections**. All criteria repeated within 
   per-brand sections; no universal criteria block.

### Edge cases the pipeline must handle gracefully

- **Crucial body areas bypass** — multiple policies allow patients to bypass 
  BSA/step requirements if hands, feet, face, neck, scalp, genitals, or 
  intertriginous areas are affected. Affects step counting logic.

- **PsA bypass for plaque step therapy** — concomitant severe PsA allows 
  skipping plaque step therapy entirely. Affects step counts.

- **Otezla classified as biologic-equivalent** — some policies treat Otezla 
  as equivalent to a biologic for step-through. May confuse branded-vs-generic 
  counting.

- **Phototherapy as denial criterion, not step** — some policies prohibit 
  concurrent phototherapy rather than requiring trial of it. Phototherapy 
  parameter must distinguish these.

- **Mislabeled documents** — at least one file (56403) has a misleading title 
  (Alaska Medicaid Tremfya) but contains a Wyoming Medicaid Stelara policy. 
  Brand input should come from submissions tab, not document title.

- **Investigational variants** — erythrodermic and guttate psoriasis listed 
  as not covered in some policies.

- **All-member step-through** — some policies apply step therapy to both new 
  starts AND continuation requests, not just initial.

### Design implications

- Step counting logic must respect bypass clauses (crucial body areas, PsA)
- Phototherapy extraction prompt must distinguish "required to try" vs 
  "prohibited from concurrent use"
- Brand detection must trust submissions tab over document filename/title

### PDF 1 (TREMFYA, Alaska Medicaid):
- 11/12 correct
- tb_test failure: "should be evaluated" in CAUTIONS interpreted as Yes; 
  should be NA. Prompt has the rule but model didn't apply it.

### PDF 2 (STELARA, Aetna Step Criteria):
- ~9-10/12 correct
- Major: preferred vs non-preferred not understood. Model attributed 
  non-preferred-access criteria to STELARA (which is preferred).
- step_therapy_text and num_steps_brands wrong.
- Fix requires prompt addition explaining the preferred-product nuance.

## Known gap: drug class membership in step counting
If v1 results show step counting failures on drug class references (e.g.,
"must fail a CAM antagonist"), add BRAND_TO_DRUG_CLASS to config.py and
pass {drug_class} as a template variable in step_therapy.xml. Deferred until
v1 evidence justifies the change.

---

## Phase 1 — 14-PDF Markdown Analysis (v1 → v2 transition)

### Document Type Taxonomy (7 confirmed types)

**Type 1 — Aetna/CVS single-drug multi-indication template**
PDFs: 378692, 324603, 176810 (Wellmark BCBS — same base template).
Structure: Coverage Criteria → Continuation of Therapy → Other (TB) →
Approval Duration and Quantity Restrictions → Quantity Level Limit.
Key quirks:
- "Authorization of N months may be granted" appears for BOTH initial
  criteria AND continuation criteria. LLM must distinguish them.
- TB test is under "## Other" with "For all indications:" preamble, not
  inside the PsO section.
- Duration labels: "Initial Approval:" / "Renewal Approval:" (NOT "initial
  authorization duration").
- QL label: "Quantity Level Limit:" (with colon and "Level" — "quantity limit"
  alone does NOT match this).

**Type 2 — Multi-indication class-level formulary**
PDFs: 365374 (ForwardHealth), 313179 (WV Medicaid).
Structure: One document, 19+ indications, all criteria at class level.
Key quirks:
- Criteria say "non-preferred cytokine and CAM antagonist drugs" not the brand.
  LLM must recognize that class-level criteria apply to the target brand when
  it appears in the non-preferred list.
- Duration in DAYS not months (183 days → 6 Months, 365 days → 12 Months).
  normalizer days branch is essential.
- Brand appears in multiple indication sections (PsO, PsA, CD, UC) — page
  filter may return cross-indication pages causing LLM confusion.
- WV Medicaid preferred/non-preferred table is an IMAGE — pdfplumber and
  docling cannot read it; accept the step count gap.

**Type 3 — Multi-drug multi-indication specialty policy**
PDFs: 362198 (HAP).
Key quirks:
- "Stelara" consistently spelled as "Stelera" — ustekinumab keyword is the fix.
- For ustekinumab/guselkumab: phototherapy is MANDATORY AND (not OR) per
  section 5a. step_phototherapy = Yes.
- TB test in "GENERAL COVERAGE CRITERIA: Applicable to all medication requests"
  — not in PsO section.
- No authorization duration specified for PsO.
- Dosing language (45mg every 12 weeks) is NOT a quantity limit — QL = NA.

**Type 4 — Large multi-drug PA+QL program**
PDFs: 326557 (BCBS/Anthem, 82p).
Key quirks:
- docling truncated to first ~20 pages (only FDA table). MarkItDown gets
  everything including the PA criteria module.
- PA criteria are in a "Module | Clinical Criteria for Approval" table
  (one large cell per module). Same criteria apply to all biologics.
- TB test explicitly in the numbered PA criteria ("patient has been tested for
  latent tuberculosis") not a separate section.
- Duration label: "Length of Approval: 12 months".
- AMJEVITA is a non-preferred adalimumab biosimilar — requires failing preferred
  agents before AMJEVITA can be approved.

**Type 5 — Drug-specific table-format policy**
PDFs: 325611 (WellSense NH Medicaid), 282478 (WellSense MA Clarity).
Key quirks:
- Criteria in markdown table. Columns: Covered Use | Exclusion Criteria |
  Required Medical Information | Coverage Duration | Other criteria.
- Duration in "Coverage Duration" column as "Initial: N months" /
  "Reauthorization: N months".
- NH Medicaid version (325611) has an additional PDL preferred agent step
  (criterion 5) not present in MA Clarity version (282478) — same template,
  different plan tiers.

**Type 6 — BRM addendum (Aetna)**
PDFs: 296961.
Key quirks:
- Non-preferred CAM requirement is in the HEADER section before numbered
  criteria: "Non-Preferred Cytokines and CAM Antagonists: Require trial and
  failure of preferred adalimumab AND one additional preferred product."
  LLM typically misses this because it focuses on numbered criteria.
- Inline age annotations per drug: "Stelara (ustekinumab) [≥6 years old]",
  "Tremfya (guselkumab)" (no annotation = adults ≥18).
  CRITICAL: LLM must extract age ONLY for the target brand and not confuse
  adjacent drug annotations.
- Duration label: "Approval Duration: 6 Months" at end of document.
- QL: "Quantity Level Limit: Reference Formulary" → NA (reference only, no
  actual limit stated).

**Type 7 — Formulary exception program**
PDFs: 258492 (AI Preferred Drug Program).
Key quirks:
- STELARA and TREMFYA are PREFERRED products. No step therapy for them.
  Step therapy criteria shown apply to NON-PREFERRED drugs.
- Exception criteria in two-column table (Preferred / Non-Preferred) with
  duplicated cell content — docling/MarkItDown produces noisy text.
- NoneType failure (LLM could not parse complex table) → extractor fix handles.

### Wrong document
PDFs: 287728 — Asuris Multiple Myeloma policy mapped to STELARA in
submissions tab. All-NA result is CORRECT. Accept and flag in ground truth.

### Critical extraction rules confirmed by markdown evidence

**Age:**
- TREMFYA plaque psoriasis FDA age is >=18 (adults only). Was incorrectly
  set to >=6 in config.py. Fixed in v2.
- For multi-brand documents (296961 BRM addendum), LLM must NOT apply
  one brand's age annotation to another brand on the same page.

**Step therapy:**
- Least restrictive path rule: if a policy has "crucial body areas" bypass
  with zero step requirements, that path exists. But the presence of a
  biologic/generic step on other paths still represents the typical requirement.
- For class-level policies: "non-preferred agents require 90-day trials of
  all preferred agents" = 1 branded step (class reference per business rules).

**Phototherapy:**
- Only Yes when phrasing is "BOTH oral agent AND phototherapy" (mandatory AND).
- Always No or NA when phototherapy is listed in OR combinations.

**TB test location:** Never inside the PsO-specific numbered criteria section.
Always in: "## Other", "GENERAL COVERAGE CRITERIA", general numbered list,
or explicitly in the numbered PA criteria (BCBS/Anthem module).

**Quantity limits:**
- Two valid label formats: "Quantity Level Limit:" (Aetna) and
  "## Quantity Limits" section header (Wellmark BCBS, WHA).
- "Reference Formulary for drug specific quantity level limits" → NA.
- Dosing schedules (e.g., "45mg every 12 weeks") → NOT a quantity limit.

**Duration:**
Seven different label formats found across payers:
  "Initial Approval: N months" (Aetna)
  "Authorization of N months may be granted" (Aetna/Wellmark)
  "Initial PA requests may be approved for up to N days" (ForwardHealth)
  "Approve for N months" (WHA 215824)
  "Coverage Duration: Initial: N months" (WellSense table format)
  "Approval Duration: N Months" (Aetna BRM addendum)
  "Length of Approval: N Month(s)" (BCBS/Anthem module)

### New prompt additions required (apply to XML files manually)
See consolidated plan in companion document. Key additions:
1. All 7 prompts: anti-fence output_format instruction
2. step_therapy.xml: class-level criteria handling; header-section step
   requirements; preferred product = no step therapy
3. duration.xml: all 7 duration label variants listed above
4. reauth.xml: "Continuation of Therapy", "Currently Receiving", "Renewal
   PA requests" as reauth section identifiers
5. tb_test.xml: check general/universal sections, not just PsO-specific
6. quantity_limits.xml: strict label rule (must say "Quantity Limit/Level
   Limit" or "QL"); dosing schedules are excluded
7. age.xml: extract age ONLY for target brand {brand}; ignore adjacent drugs
8. step_phototherapy.xml: AND override detection within subsections