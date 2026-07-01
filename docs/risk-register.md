# Risk Register

**Project:** AI-Assisted Emergency Department Triage Pilot  
**Organisation:** Mercer General Hospital

This risk register identifies the principal AI-technical, operational, ethical, and equity risks associated with the proposed AI-assisted emergency department triage pilot. Each risk includes its likelihood, impact, mitigation strategy, and a measurable signal of success.

---

## Risk 1 – Training-distribution shift between Mercer urban ED population and secondary deployment sites

**Category:** AI-Technical

**Likelihood:** High (H)

**Impact:** High (H)

**Mitigation:**  
Conduct prospective external validation on a held-out dataset from each new deployment site before go-live. Compare Level 1–2 sensitivity against the Mercer baseline. Halt rollout to that site if the sensitivity gap exceeds three percentage points and initiate retraining using local data.

**Signal of Success:**  
External validation confirms Level 1–2 sensitivity at each new site is within ±3 percentage points of Mercer training-site performance before deployment. Site-level performance is reviewed and documented quarterly thereafter.

---

## Risk 2 – Vital-sign input incompleteness producing false-low acuity recommendations during documentation lag periods

**Category:** AI-Technical

**Likelihood:** High (H)

**Impact:** High (H)

**Mitigation:**  
Gate acuity score generation so that if more than two vital-sign fields are absent, the system displays only the data-completeness panel and withholds the acuity recommendation until sufficient data are recorded. Timestamp every recommendation with a data-completeness indicator showing which fields were available when the recommendation was generated.

**Signal of Success:**  
Weekly electronic health record audits confirm zero acuity scores were generated with more than two missing vital-sign fields. The data-completeness indicator appears on 100% of recommendations throughout the audit period.

---

## Risk 3 – Alert fatigue from sustained high false-positive rates causing nurses to dismiss AI escalation alerts

**Category:** Operational

**Likelihood:** High (H)

**Impact:** High (H)

**Mitigation:**  
Configure the alert threshold before deployment so that the positive predictive value for high-acuity alerts reaches at least 40%. Monitor the alert-to-clinical-action rate weekly, report findings to the named clinical owner, and adjust thresholds whenever sustained performance falls below the expected range.

**Signal of Success:**  
The alert-to-clinical-action rate remains at or above 50% across monitored shifts. Any two consecutive weeks below 40% trigger a mandatory clinical governance review with documented threshold adjustments within seven working days.

---

## Risk 4 – EHR integration failure leaving triage nurses without AI support and without a tested manual fallback protocol

**Category:** Operational

**Likelihood:** Low (L)

**Impact:** Medium (M)

**Mitigation:**  
Develop and document a manual triage protocol before deployment. Train all triage-qualified staff in both AI-assisted and manual workflows, and conduct quarterly downtime simulation drills.

**Signal of Success:**  
All triage-qualified staff complete the manual fallback drill before go-live and annually thereafter. Drill completion records are audited quarterly by the clinical governance lead, with performance reviewed during governance meetings.

---

## Risk 5 – Automation bias causing triage nurses to accept AI recommendations without independent clinical assessment

**Category:** Ethical

**Likelihood:** Medium (M)

**Impact:** High (H)

**Mitigation:**  
Embed override documentation into pre-deployment training. Require nurses to complete structured simulation exercises in which deliberately incorrect AI recommendations must be identified and overridden. Monitor live override rates against a predefined expected range.

**Signal of Success:**  
Staff correctly override incorrect AI recommendations in at least 90% of simulation scenarios during six- and twelve-month assessments. Live override rates remain between 5% and 20%. A sustained drop below 5% across any three-week period triggers mandatory retraining.

---

## Risk 6 – Absence of patient disclosure that AI contributed to triage prioritisation, creating an informed consent gap

**Category:** Ethical

**Likelihood:** Medium (M)

**Impact:** Medium (M)

**Mitigation:**  
Develop a plain-language patient disclosure explaining that an AI decision-support tool assists triage nurses. Obtain review from a patient advocacy group and legal counsel before deployment. Include the disclosure during registration and at the triage desk.

**Signal of Success:**  
Patient disclosure language receives formal approval before go-live. Three-month patient satisfaction surveys include an AI-disclosure comprehension question, with at least 70% of respondents demonstrating understanding.

---

## Risk 7 – Systematic under-triage of elderly patients (>65 years) with atypical high-acuity presentations

**Category:** Equity

**Likelihood:** Medium (M)

**Impact:** High (H)

**Mitigation:**  
Audit training-data demographics before model development. If patients older than 65 years comprise fewer than 15% of training cases, oversample Mercer records or supplement the dataset using de-identified retrospective data from comparable Caribbean hospitals. Report sensitivity and specificity separately by age group during validation.

**Signal of Success:**  
Quarterly subgroup audits confirm Level 1–2 sensitivity for patients older than 65 years remains within five percentage points of overall model sensitivity. Any sustained gap greater than five percentage points triggers mandatory model retraining, with findings reported through clinical governance.

---

## Risk 8 – Infrastructure disparity during rollout to lower-resource Caribbean hospitals

**Category:** Equity

**Likelihood:** Medium (M)

**Impact:** Medium (M)

**Mitigation:**  
Develop a mandatory site-readiness checklist covering electronic health record compatibility, hardware requirements, connectivity reliability, and power backup. Require sign-off by both the local IT lead and clinical owner before deployment. Maintain documented offline fallback procedures for sites with intermittent connectivity.

**Signal of Success:**  
Site-readiness checklists are completed before each deployment. Monthly monitoring confirms at least 98% AI system availability at each deployed site. Any site falling below this threshold receives a documented infrastructure improvement plan within 30 days.

---

## Summary

| Category | Number of Risks | Highest Impact |
|----------|----------------:|:--------------:|
| AI-Technical | 2 | High |
| Operational | 2 | High |
| Ethical | 2 | High |
| Equity | 2 | High |
