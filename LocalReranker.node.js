/**
 * ======================================================================
 * 🔹 Local Hybrid Reranker (v2.1 — release-ready)
 * ----------------------------------------------------------------------
 * Offline, privacy-first hybrid reranker for n8n:
 *  - Semantic (cosine similarity) or external score
 *  - Lexical (BM25 + exact token overlap)
 *  - Recency boost (configurable half-life)
 *  - Dynamic BM25 normalization: sigmoid (default) | min–max per batch
 *  - Global DF for proper BM25 IDF
 *  - Custom stopwords, learned logistic model (optional)
 *  - Batch-safe, debug-friendly, weight validation (soft warning)
 * ======================================================================
 */

class LocalReranker {
  constructor() {
    this.description = {
      displayName: "Local Reranker (v2.1)",
      name: "localReranker",
      group: ["transform"],
      version: 1,
      description: "Offline hybrid reranker (semantic + lexical + recency) with advanced options",
      defaults: { name: "Local Reranker" },
      inputs: ["main"],
      outputs: ["main"],
      properties: [
        // Core
        { displayName: "Top N", name: "topN", type: "number", default: 5 },
        { displayName: "Batch Size", name: "batchSize", type: "number", default: 200, description: "Process documents in batches (for large corpora)" },

        // Weights (with soft validation)
        { displayName: "Semantic Weight", name: "semWeight", type: "number", default: 0.7 },
        { displayName: "Lexical Weight", name: "lexWeight", type: "number", default: 0.25 },
        { displayName: "Exact Match Weight", name: "exactWeight", type: "number", default: 0.10 },
        { displayName: "Recency Weight", name: "timeWeight", type: "number", default: 0.05 },

        // Options
        { displayName: "Use External Score", name: "useExternalScore", type: "boolean", default: false, description: "Use it.json.score instead of cosine similarity" },
        { displayName: "Recency Half-Life (days)", name: "recencyHalfLife", type: "number", default: 60 },
        { displayName: "Min Token Length", name: "minTokenLength", type: "number", default: 2 },

        {
          displayName: "Normalization Method",
          name: "normalizeMethod",
          type: "options",
          options: [
            { name: "Sigmoid (stable default)", value: "sigmoid" },
            { name: "Min–Max per batch (adaptive)", value: "minmax" }
          ],
          default: "sigmoid",
          description: "How to normalize BM25 into [0,1]"
        },

        { displayName: "Custom Stopwords (comma-separated)", name: "customStopwords", type: "string", default: "", description: "Extra stopwords to ignore (comma-separated, case-insensitive)" },

        // Learned model (optional)
        { displayName: "Use Learned Model", name: "useModel", type: "boolean", default: false, description: "Use logistic regression with provided weights" },
        { displayName: "Model Weights (sem,bm25,exact,rec,bias)", name: "modelWeights", type: "string", default: "", description: "Example: 0.8,0.5,0.3,0.1,-0.2 (trained on your data)" },

        // Debug
        { displayName: "Debug Mode", name: "debug", type: "boolean", default: false, description: "Outputs detailed scoring breakdown per document" }
      ]
    };
  }

  async execute() {
    try {
      const items = this.getInputData();
      if (!items?.length) throw new Error("No input items provided.");

      // Params
      const topN = Number(this.getNodeParameter("topN", 0)) || 5;
      const batchSize = Number(this.getNodeParameter("batchSize", 0)) || 200;
      const useExternalScore = Boolean(this.getNodeParameter("useExternalScore", 0));
      const recencyHalfLife = Number(this.getNodeParameter("recencyHalfLife", 0)) || 60;
      const minTokenLength = Number(this.getNodeParameter("minTokenLength", 0)) || 2;
      const debug = Boolean(this.getNodeParameter("debug", 0));
      const normalizeMethod = String(this.getNodeParameter("normalizeMethod", 0) || "sigmoid");

      const W = {
        sem: Number(this.getNodeParameter("semWeight", 0)) || 0.7,
        lex: Number(this.getNodeParameter("lexWeight", 0)) || 0.25,
        exact: Number(this.getNodeParameter("exactWeight", 0)) || 0.10,
        time: Number(this.getNodeParameter("timeWeight", 0)) || 0.05
      };
      // Soft validation of weights sum (~1.0)
      const sumW = W.sem + W.lex + W.exact + W.time;
      if (sumW < 0.9 || sumW > 1.1) {
        const msg = `⚠️ Weights sum to ${sumW.toFixed(2)} — recommended total ≈ 1.0`;
        if (this.logger?.warn) this.logger.warn(msg); else console.warn(msg);
      }

      const useModel = Boolean(this.getNodeParameter("useModel", 0));
      const modelWeightsStr = String(this.getNodeParameter("modelWeights", 0) || "");

      // Input validation
      const first = items[0].json || {};
      if (typeof first.query !== "string") throw new Error("Missing or invalid 'query'.");
      if (!Array.isArray(first.query_embedding)) throw new Error("Missing or invalid 'query_embedding'.");
      if (!Array.isArray(first.items)) throw new Error("Missing or invalid 'items' array.");

      const query = first.query.toLowerCase().trim();
      const qEmb = first.query_embedding || [];
      const docs = first.items || [];

      // Parse custom stopwords
      const customStop = String(this.getNodeParameter("customStopwords", 0) || "")
        .split(",").map(s => s.trim().toLowerCase()).filter(Boolean);

      // Helpers
      const cos = (a, b) => {
        if (!a?.length || !b?.length || a.length !== b.length) return 0;
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
        const denom = Math.sqrt(na) * Math.sqrt(nb);
        return denom ? dot / denom : 0;
      };

      const BASE_STOP = [
        "the","is","at","on","a","an","and","or",
        "и","на","по","для","как","это","не","из","та","що","але","від"
      ];
      const STOP = new Set([...BASE_STOP, ...customStop]);

      const tok = t =>
        (t || "")
          .toLowerCase()
          .normalize("NFC")
          .replace(/[^\p{L}\p{N}]+/gu, " ")
          .split(/\s+/)
          .filter(w => w.length >= minTokenLength && !STOP.has(w));

      const qTok = tok(query);
      if (!qTok.length || !docs.length) return [this.helpers.returnJsonArray([])];

      // BM25 constants
      const k1 = 1.2, b = 0.75;
      const now = Date.now();
      const results = [];

      // Optional: parse model weights
      let modelWeights = null;
      if (useModel) {
        const arr = modelWeightsStr.split(",").map(x => Number(x.trim())).filter(x => !Number.isNaN(x));
        if (arr.length !== 5) throw new Error("Model weights must contain exactly 5 numbers: sem,bm25,exact,rec,bias");
        modelWeights = { wSem: arr[0], wBm25: arr[1], wExact: arr[2], wRec: arr[3], bias: arr[4] };
      }

      // --- Global DF (document frequency across the whole corpus) ---
      const globalDF = {};
      docs.forEach(it => {
        const set = new Set(tok(it.json?.chunk || ""));
        qTok.forEach(t => { if (set.has(t)) globalDF[t] = (globalDF[t] || 0) + 1; });
      });

      // Batch processing
      for (let i = 0; i < docs.length; i += batchSize) {
        const batch = docs.slice(i, i + batchSize);

        // Precompute avg length for this batch (BM25 length normalization)
        const avgLen = batch.reduce((s, d) => s + tok(d.json?.chunk || "").length, 0) / Math.max(batch.length, 1);

        // First pass: compute raw metrics and cache bm25 for normalization
        const pre = batch.map(it => {
          const txt = it.json?.chunk || "";
          const dTok = tok(txt);
          const dSet = new Set(dTok);
          const len = dTok.length || 1;

          const sem = useExternalScore
            ? (typeof it.json?.score === "number" ? it.json.score : 0)
            : (qEmb.length && it.json?.embedding?.length ? cos(qEmb, it.json.embedding) : 0);

          const tfMap = {};
          dTok.forEach(t => (tfMap[t] = (tfMap[t] || 0) + 1));

          let bm25 = 0;
          qTok.forEach(t => {
            const tf = tfMap[t] || 0;
            if (tf) {
              // IDF by global DF, not batch
              const idf = Math.log((docs.length + 1) / (globalDF[t] || 1));
              bm25 += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len / avgLen))));
            }
          });

          const exact = qTok.filter(t => dSet.has(t)).length / qTok.length;

          const ts = it.json?.timestamp;
          const docTime = ts ? new Date(ts).getTime() : NaN;
          const days = Number.isNaN(docTime) ? 999 : (now - docTime) / 86400000;
          const rec = Math.exp(-days / recencyHalfLife);

          return { it, sem, bm25, exact, rec };
        });

        // Build BM25 normalizer
        let normBM25;
        if (normalizeMethod === "minmax") {
          const vals = pre.map(p => p.bm25);
          const minBM25 = Math.min(...vals);
          const maxBM25 = Math.max(...vals);
          normBM25 = (v) => (maxBM25 === minBM25) ? 0 : (v - minBM25) / (maxBM25 - minBM25);
        } else {
          // sigmoid default
          normBM25 = (v) => 1 / (1 + Math.exp(-v));
        }

        // Second pass: final score
        pre.forEach(p => {
          let final;
          if (useModel && modelWeights) {
            // logistic regression
            const z = p.sem * modelWeights.wSem
              + normBM25(p.bm25) * modelWeights.wBm25
              + p.exact * modelWeights.wExact
              + p.rec * modelWeights.wRec
              + modelWeights.bias;
            final = 1 / (1 + Math.exp(-z));
          } else {
            // weighted hybrid
            final = (p.sem * W.sem)
              + (normBM25(p.bm25) * W.lex)
              + (p.exact * W.exact)
              + (p.rec * W.time);
          }

          const out = {
            ...p.it,
            json: {
              ...p.it.json,
              rerank_score: final,
              bm25_raw: p.bm25
            }
          };
          if (debug) out.json.debug = { sem: p.sem, bm25: p.bm25, exact: p.exact, rec: p.rec, bm25n_method: normalizeMethod, weights: { ...W, sum: sumW } };
          results.push(out);
        });
      }

      // Sort & slice
      const sorted = results.sort((a, b) => b.json.rerank_score - a.json.rerank_score).slice(0, topN);
      return [this.helpers.returnJsonArray(sorted)];

    } catch (err) {
      return [this.helpers.returnJsonArray([{ error: true, message: err.message || "Unknown reranker error" }])];
    }
  }
}

module.exports = { nodeClass: LocalReranker };
