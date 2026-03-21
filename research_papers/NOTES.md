# Research Notes — VEGA

A synthesis of the five papers that shaped VEGA's methodology,
with explicit notes on what was adopted, what was improved upon,
and what gaps VEGA addresses.

---

## 1. Twitter Mood Predicts the Stock Market
**Bollen, Mao & Zeng (2011) — Journal of Computational Science**

**Core finding:** Collective public mood measured from Twitter feeds is
causally (Granger) predictive of DJIA movements, with 86.7% directional
accuracy. Crucially, not all mood dimensions predict equally — "Calm" was
the strongest predictor while "Happy" was not significant.

**What VEGA borrows:**
- Multi-dimensional sentiment scoring instead of a single polarity score.
  VEGA preserves FinBERT's three output dimensions (positive%, negative%,
  uncertainty%) as separate features in the composite signal rather than
  collapsing them — directly inspired by this paper's finding that
  dimension choice matters.
- Granger causality framing: VEGA's backtest tests whether sentiment
  *leads* returns, not just correlates with them.

**Where VEGA advances:**
- Source is earnings call transcripts (structured, company-specific,
  high-signal) vs. Twitter feeds (noisy, general public).
- Uses contextual transformer embeddings (FinBERT) vs. dictionary-based
  OpinionFinder/GPOMS.
- India market focus vs. DJIA.

---

## 2. Detecting Deceptive Discussions in Conference Calls
**Larcker & Zakolyukina (2012) — Journal of Accounting Research**

**Core finding:** Linguistic features of CEO/CFO narratives in earnings
calls predict subsequent financial restatements. Deceptive executives use
more extreme positive emotion, fewer anxiety words, fewer shareholder value
references, and more general knowledge references. Out-of-sample models
beat random chance by 6–16%.

**What VEGA borrows:**
- Linguistic risk flag categories: extreme positive emotion (overconfidence
  signal), anxiety word frequency (stress signal), shareholder value
  references (deflection signal), vague/general knowledge language
  (evasion signal). Implemented in VEGA's `risk_flagger.py` using the
  Loughran-McDonald financial word list alongside MiniLM embeddings.
- The MD&A vs. Q&A distinction: VEGA separates prepared remarks from
  Q&A sections where possible, as Larcker & Zakolyukina find CFO
  Q&A narratives are the strongest deception signal.
- Portfolio alpha result (-4% to -11% annualized for high-deception
  firms) provides a benchmark for VEGA's backtest interpretation.

**Where VEGA advances:**
- Uses contextual BERT embeddings vs. bag-of-words word category counts.
  Contextual models capture negation, hedging, and sentence-level meaning
  that word lists miss entirely.
- Applied to Indian NIFTY 50 companies — no prior study applies these
  linguistic deception features to Indian corporate communications.
- Combines linguistic risk flags with quantitative sentiment scoring into
  a unified composite signal.

---

## 3. Predictability of Earnings and Its Impact on Stock Returns: Evidence from India
**Kundu & Banerjee (2021) — Cogent Economics & Finance**

**Core finding:** Using 67 large-cap Indian stocks over 33 quarters
(2010–2018), all stocks experience return premiums pre-announcement, but
firms reporting better earnings generate significantly higher returns.
Direction of earnings change matters more than magnitude. Year-on-year
comparisons are more informative than sequential quarter comparisons.
Evidence of information leakage in Indian markets.

**What VEGA borrows:**
- Earnings variables as OLS controls: REV (revenue), PBDIT, EFFO (earnings
  from core operations), and PAT are included as control variables in
  VEGA's regression so that the NLP composite signal is tested for
  *incremental* explanatory power above fundamental information.
- Year-on-year benchmark: VEGA uses year-on-year abnormal returns as the
  primary dependent variable, consistent with this paper's finding that
  YoY comparisons are more market-relevant than QoQ.
- Pre-announcement window: VEGA's abnormal return window includes the
  3-day pre-announcement period, not just post-announcement.

**Where VEGA advances:**
- Adds the linguistic layer entirely absent from Kundu & Banerjee.
  The hypothesis is that management tone in concalls contains predictive
  information *beyond* what's in the financial numbers themselves.
- More recent data (post-2021) and specifically targets concall language
  rather than numerical earnings variables alone.

---

## 4. Sentiment Analysis Models for Bank Nifty Index
**Ukhalkar & Zirmite (2023) — ICCUBEA Conference**

**Core finding:** Survey of sentiment analysis approaches on Indian market
data. Models reviewed include VADER, TextBlob, SVM, and basic LSTM
approaches applied to news and social media for Bank Nifty prediction.
Identifies research gaps in domain-specific and deeper NLP approaches.

**What VEGA borrows:**
- Explicit acknowledgment of the gap: this paper confirms that existing
  Indian market NLP work is limited to shallow models (VADER, TextBlob)
  on social media/news data. No prior work applies transformer-based
  contextual models to Indian earnings call transcripts specifically.
  VEGA directly addresses this gap.

**Where VEGA advances:**
- FinBERT (finance-domain BERT) vs. VADER/TextBlob — order-of-magnitude
  improvement in financial language understanding.
- Earnings concall transcripts vs. news/social media — higher signal-to-
  noise ratio, structured source, company-specific.
- Quantitative backtest vs. qualitative model comparison.

---

## 5. FinBERT2: A Specialized Bidirectional Encoder for Finance
**Xu, Wen et al. (2025) — KDD '25**

**Core finding:** FinBERT2, pretrained on 32B tokens of financial text,
outperforms original FinBERT variants by 0.4–3.3% and leading LLMs by
9.7–12.3% on financial classification tasks. Fine-tuned BERT-style
discriminative models consistently outperform generative LLMs on
classification and retrieval tasks in finance, even in the LLM era.

**What VEGA borrows:**
- Validates the choice of fine-tuned BERT-style model (FinBERT) over
  prompting an LLM for classification tasks. VEGA uses FinBERT for
  sentiment scoring and reserves Gemini Flash only for narrative
  *generation* — consistent with FinBERT2's finding that each model
  class has a distinct role.
- FinBERT-FLS (Forward-Looking Statement classifier, yya518/FinBERT):
  VEGA uses this variant specifically for forward guidance extraction,
  replacing the BART zero-shot classifier in the guidance detection step.
  FinBERT-FLS is fine-tuned on analyst reports for exactly this task.

**Where VEGA advances:**
- FinBERT2 itself is not yet publicly available as a HuggingFace
  loadable model. VEGA uses the best currently available alternative
  (ProsusAI/FinBERT + yya518/FinBERT-FLS) and will upgrade to FinBERT2
  when weights are released — noted as future work.

---

## Summary: VEGA's research contribution

Every prior paper reviewed uses at least one of: bag-of-words models,
social media data, US/global markets, or numerical-only features.

VEGA's combination is novel along four dimensions simultaneously:
1. **Contextual transformer NLP** (FinBERT) vs. dictionary/bag-of-words
2. **Earnings concall transcripts** vs. news/social media
3. **Indian NIFTY 50 market** vs. US-centric prior work
4. **Composite signal** combining sentiment + linguistic risk flags +
   guidance classification + fundamental controls in one regression

No single prior paper combines all four. That is VEGA's contribution.
