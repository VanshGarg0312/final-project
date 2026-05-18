import { useState, useRef } from "react";

const API = "http://localhost:8000";

const SPRING = "http://localhost:8080/api";

async function fetchPredict(text) {
  const res = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

async function fetchFeedback(text, predicted_label) {
  const res = await fetch(`${API}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, predicted_label }),
  });
  return res.json();
}

// ── SHAP Bar ─────────────────────────────────────────────────────────
function ShapBar({ word, importance, max }) {
  const positive = importance > 0;
  const pct = Math.min((Math.abs(importance) / max) * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <span style={{
        width: 120, fontSize: 13, fontFamily: "var(--font-mono)",
        color: "var(--color-text-secondary)", textAlign: "right",
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap"
      }}>{word}</span>
      <div style={{ flex: 1, height: 10, background: "var(--color-background-secondary)", borderRadius: 5, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: positive ? "#1D9E75" : "#D85A30",
          borderRadius: 5,
          transition: "width 0.6s cubic-bezier(.4,0,.2,1)"
        }} />
      </div>
      <span style={{
        width: 54, fontSize: 12, fontFamily: "var(--font-mono)",
        color: positive ? "#0F6E56" : "#993C1D",
        textAlign: "right"
      }}>{importance > 0 ? "+" : ""}{importance.toFixed(3)}</span>
    </div>
  );
}

// ── Confidence Ring ───────────────────────────────────────────────────
function ConfidenceRing({ value, label, color }) {
  const r = 36, circ = 2 * Math.PI * r;
  const dash = circ * value;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <svg width={90} height={90} viewBox="0 0 90 90">
        <circle cx={45} cy={45} r={r} fill="none" stroke="var(--color-background-secondary)" strokeWidth={7} />
        <circle cx={45} cy={45} r={r} fill="none"
          stroke={color} strokeWidth={7}
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 45 45)"
          style={{ transition: "stroke-dasharray 0.8s cubic-bezier(.4,0,.2,1)" }}
        />
        <text x={45} y={49} textAnchor="middle" fontSize={15} fontWeight={500}
          fill="var(--color-text-primary)" fontFamily="var(--font-sans)">
          {Math.round(value * 100)}%
        </text>
      </svg>
      <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{label}</span>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────
export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const textareaRef = useRef();

  const isFake = result?.prediction === "FAKE";
  const accentColor = isFake ? "#D85A30" : "#1D9E75";
  const accentBg = isFake ? "#FAECE7" : "#E1F5EE";
  const accentText = isFake ? "#993C1D" : "#0F6E56";

  const maxShap = result?.shap?.length
    ? Math.max(...result.shap.map(s => Math.abs(s.importance)))
    : 1;

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setFeedbackSent(false);
    try {
      const data = await fetchPredict(text.trim());
      setResult(data);
    } catch (e) {
      setError("Could not connect to ML service. Make sure it's running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  async function handleWrong() {
    if (!result) return;
    setFeedbackLoading(true);
    try {
      await fetchFeedback(text.trim(), result.prediction);
      setFeedbackSent(true);
    } catch {
      setFeedbackSent(true);
    } finally {
      setFeedbackLoading(false);
    }
  }

  function handleClear() {
    setText("");
    setResult(null);
    setError(null);
    setFeedbackSent(false);
    textareaRef.current?.focus();
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--color-background-tertiary)",
      fontFamily: "var(--font-sans)",
      padding: "2.5rem 1rem",
    }}>

      {/* ── Header ── */}
      <div style={{ maxWidth: 720, margin: "0 auto 2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "#26215C", display: "flex", alignItems: "center", justifyContent: "center"
          }}>
            <svg width={20} height={20} viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth={2} strokeLinecap="round">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 500, color: "var(--color-text-primary)", letterSpacing: "-0.3px" }}>
              Fake News Detector
            </h1>
            <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-secondary)" }}>
              Explainable AI · Logistic Regression + SHAP
            </p>
          </div>
        </div>
      </div>

      {/* ── Input Card ── */}
      <div style={{
        maxWidth: 720, margin: "0 auto 1.5rem",
        background: "var(--color-background-primary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "1.25rem",
      }}>
        <label style={{ fontSize: 13, color: "var(--color-text-secondary)", display: "block", marginBottom: 8 }}>
          Paste news article or headline
        </label>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Enter the news text you want to verify…"
          rows={6}
          style={{
            width: "100%", boxSizing: "border-box",
            resize: "vertical", fontFamily: "var(--font-sans)",
            fontSize: 15, lineHeight: 1.6,
            padding: "10px 12px",
            border: "0.5px solid var(--color-border-secondary)",
            borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-secondary)",
            color: "var(--color-text-primary)",
            outline: "none",
          }}
          onKeyDown={e => { if (e.ctrlKey && e.key === "Enter") handleAnalyze(); }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12 }}>
          <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
            {text.length} chars · Ctrl+Enter to analyze
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            {text && (
              <button onClick={handleClear} style={{
                padding: "7px 14px", fontSize: 13,
                borderRadius: "var(--border-radius-md)",
                border: "0.5px solid var(--color-border-secondary)",
                background: "transparent", color: "var(--color-text-secondary)",
                cursor: "pointer"
              }}>Clear</button>
            )}
            <button
              onClick={handleAnalyze}
              disabled={!text.trim() || loading}
              style={{
                padding: "7px 20px", fontSize: 14, fontWeight: 500,
                borderRadius: "var(--border-radius-md)",
                border: "none",
                background: !text.trim() || loading ? "var(--color-background-secondary)" : "#26215C",
                color: !text.trim() || loading ? "var(--color-text-tertiary)" : "#fff",
                cursor: !text.trim() || loading ? "not-allowed" : "pointer",
                transition: "background 0.2s",
              }}
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>
          </div>
        </div>
      </div>

      {/* ── Error ── */}
      {error && (
        <div style={{
          maxWidth: 720, margin: "0 auto 1.5rem",
          background: "var(--color-background-danger)",
          border: "0.5px solid var(--color-border-danger)",
          borderRadius: "var(--border-radius-md)",
          padding: "12px 16px", fontSize: 14,
          color: "var(--color-text-danger)"
        }}>{error}</div>
      )}

      {/* ── Loading Skeleton ── */}
      {loading && (
        <div style={{ maxWidth: 720, margin: "0 auto" }}>
          {[140, 80, 200].map((h, i) => (
            <div key={i} style={{
              height: h, borderRadius: "var(--border-radius-lg)",
              background: "var(--color-background-secondary)",
              marginBottom: 12,
              animation: "pulse 1.4s ease-in-out infinite",
              animationDelay: `${i * 0.15}s`
            }} />
          ))}
          <style>{`@keyframes pulse { 0%,100%{opacity:.6} 50%{opacity:1} }`}</style>
        </div>
      )}

      {/* ── Result ── */}
      {result && !loading && (
        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Verdict Card */}
          <div style={{
            background: "var(--color-background-primary)",
            border: `2px solid ${accentColor}`,
            borderRadius: "var(--border-radius-lg)",
            padding: "1.25rem 1.5rem",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexWrap: "wrap", gap: 16
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{
                width: 48, height: 48, borderRadius: 12,
                background: accentBg,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 22
              }}>
                {isFake ? "⚠️" : "✅"}
              </div>
              <div>
                <div style={{
                  fontSize: 26, fontWeight: 500, letterSpacing: "-0.5px",
                  color: accentText
                }}>
                  {result.prediction}
                </div>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 2 }}>
                  {result.source === "feedback"
                    ? "Based on your past feedback correction"
                    : "Model prediction"}
                </div>
              </div>
            </div>

            <ConfidenceRing
              value={result.confidence}
              label="Confidence"
              color={accentColor}
            />
          </div>

          {/* Feedback banner from feedback source */}
          {result.source === "feedback" && (
            <div style={{
              background: "var(--color-background-info)",
              border: "0.5px solid var(--color-border-info)",
              borderRadius: "var(--border-radius-md)",
              padding: "10px 14px", fontSize: 13,
              color: "var(--color-text-info)",
              display: "flex", alignItems: "center", gap: 8
            }}>
              <svg width={16} height={16} viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth={2} strokeLinecap="round">
                <circle cx={12} cy={12} r={10} />
                <line x1={12} y1={8} x2={12} y2={12} />
                <line x1={12} y1={16} x2={12.01} y2={16} />
              </svg>
              This result was retrieved from your feedback history — not the live model.
            </div>
          )}

          {/* SHAP Explanation Card */}
          {result.shap?.length > 0 && (
            <div style={{
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-lg)",
              padding: "1.25rem 1.5rem",
            }}>
              <div style={{ marginBottom: 16 }}>
                <h2 style={{ margin: 0, fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)" }}>
                  Why this prediction?
                </h2>
                <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--color-text-secondary)" }}>
                  <span style={{
                    display: "inline-block", width: 10, height: 10,
                    background: "#1D9E75", borderRadius: 2, marginRight: 5
                  }} />
                  pushes toward REAL &nbsp;·&nbsp;
                  <span style={{
                    display: "inline-block", width: 10, height: 10,
                    background: "#D85A30", borderRadius: 2, marginRight: 5
                  }} />
                  pushes toward FAKE
                </p>
              </div>
              {result.shap.map((s, i) => (
                <ShapBar key={i} word={s.word} importance={s.importance} max={maxShap} />
              ))}
            </div>
          )}

          {/* Feedback Card */}
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "1rem 1.5rem",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexWrap: "wrap", gap: 12
          }}>
            <div>
              <p style={{ margin: 0, fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>
                Is this result wrong?
              </p>
              <p style={{ margin: "2px 0 0", fontSize: 13, color: "var(--color-text-secondary)" }}>
                Your correction will be saved and used to improve future predictions.
              </p>
            </div>
            {feedbackSent ? (
              <div style={{
                fontSize: 13, color: "var(--color-text-success)",
                background: "var(--color-background-success)",
                padding: "7px 14px", borderRadius: "var(--border-radius-md)",
                border: "0.5px solid var(--color-border-success)"
              }}>
                ✓ Feedback saved
              </div>
            ) : (
              <button
                onClick={handleWrong}
                disabled={feedbackLoading}
                style={{
                  padding: "7px 16px", fontSize: 13,
                  borderRadius: "var(--border-radius-md)",
                  border: "0.5px solid var(--color-border-danger)",
                  background: "var(--color-background-danger)",
                  color: "var(--color-text-danger)",
                  cursor: feedbackLoading ? "not-allowed" : "pointer",
                  fontFamily: "var(--font-sans)",
                }}
              >
                {feedbackLoading ? "Saving…" : `Mark as ${isFake ? "REAL" : "FAKE"}`}
              </button>
            )}
          </div>

        </div>
      )}
    </div>
  );
}