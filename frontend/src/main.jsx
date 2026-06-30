import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { FileText, Upload, Search, GitCompareArrows, Loader2 } from "lucide-react";
import "./styles.css";

const apiBase = import.meta.env.VITE_API_BASE || "http://localhost:8080/api";

async function api(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Request failed");
  }
  return response.json();
}

function Source({ source }) {
  if (!source) return null;
  return <span className="source">{source.section} p.{source.page}</span>;
}

function EvidenceList({ title, items }) {
  if (!items?.length) return null;
  return (
    <section className="panel">
      <h2>{title}</h2>
      <div className="evidence-list">
        {items.map((item, index) => (
          <article className="evidence" key={`${title}-${index}`}>
            <p>{item.text}</p>
            <Source source={item.source} />
          </article>
        ))}
      </div>
    </section>
  );
}

function EvidenceStack({ items }) {
  if (!items?.length) return null;
  return (
    <div className="evidence-list">
      {items.map((item, index) => (
        <article className="evidence" key={index}>
          <p>{item.text}</p>
          <Source source={item.source} />
        </article>
      ))}
    </div>
  );
}

function UploadBox({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    const form = event.currentTarget;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const paper = await api("/papers", { method: "POST", body });
      onUploaded(paper);
      setFile(null);
      form.reset();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="upload" onSubmit={submit}>
      <label>
        <FileText size={18} />
        <span>{file?.name || "Choose a research paper PDF"}</span>
        <input type="file" accept="application/pdf" onChange={(event) => setFile(event.target.files?.[0])} />
      </label>
      <button disabled={!file || busy}>
        {busy ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
        Upload
      </button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}

function App() {
  const [papers, setPapers] = useState([]);
  const [selected, setSelected] = useState("");
  const [summary, setSummary] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [compareIds, setCompareIds] = useState(["", ""]);
  const [comparison, setComparison] = useState(null);
  const [busy, setBusy] = useState("");
  const selectedPaper = useMemo(() => papers.find((paper) => paper.id === selected), [papers, selected]);

  async function loadPapers() {
    const data = await api("/papers");
    setPapers(data);
    if (!selected && data[0]) setSelected(data[0].id);
  }

  useEffect(() => {
    loadPapers().catch(() => setPapers([]));
  }, []);

  async function loadSummary(id = selected) {
    if (!id) return;
    setBusy("summary");
    setSummary(null);
    try {
      setSummary(await api(`/papers/${id}/summary`));
    } finally {
      setBusy("");
    }
  }

  async function ask(event) {
    event.preventDefault();
    if (!selected || !question.trim()) return;
    setBusy("qa");
    setAnswer(null);
    try {
      setAnswer(await api(`/papers/${selected}/qa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }));
    } finally {
      setBusy("");
    }
  }

  async function compare(event) {
    event.preventDefault();
    if (!compareIds[0] || !compareIds[1]) return;
    setBusy("compare");
    setComparison(null);
    try {
      setComparison(await api("/papers/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paperIds: compareIds }),
      }));
    } finally {
      setBusy("");
    }
  }

  function addUploaded(paper) {
    setPapers((current) => [paper, ...current.filter((item) => item.id !== paper.id)]);
    setSelected(paper.id);
    setSummary(null);
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <p className="eyebrow">RAG-Based Research Paper Claim Summarizer</p>
          <h1>ResearchBuddy AI</h1>
        </div>
        <select value={selected} onChange={(event) => setSelected(event.target.value)}>
          <option value="">Select paper</option>
          {papers.map((paper) => (
            <option key={paper.id} value={paper.id}>{paper.title}</option>
          ))}
        </select>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <UploadBox onUploaded={addUploaded} />
          <section className="panel compact">
            <h2>Papers</h2>
            <div className="paper-list">
              {papers.map((paper) => (
                <button className={paper.id === selected ? "active paper" : "paper"} key={paper.id} onClick={() => setSelected(paper.id)}>
                  <FileText size={16} />
                  <span>{paper.title}</span>
                </button>
              ))}
            </div>
          </section>
          <section className="panel compact">
            <h2>Ask a question</h2>
            <form className="qa" onSubmit={ask}>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What claim is supported by the results?" />
              <button disabled={!selected || !question.trim() || busy === "qa"}>
                {busy === "qa" ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
                Ask
              </button>
            </form>
          </section>
        </aside>

        <section className="content">
          <div className="hero">
            <div>
              <p className="eyebrow">Current paper</p>
              <h2>{selectedPaper?.title || "Upload a PDF to begin"}</h2>
            </div>
            <button disabled={!selected || busy === "summary"} onClick={() => loadSummary()}>
              {busy === "summary" ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
              Generate Summary
            </button>
          </div>

          {answer && (
            <section className="panel answer">
              <h2>Grounded Answer</h2>
              <p>{answer.answer}</p>
              <div className="sources">
                {answer.sources?.map((source, index) => <Source key={index} source={source} />)}
              </div>
            </section>
          )}

          {summary && (
            <div className="summary-grid">
              <EvidenceList title="Overview" items={summary.overview} />
              <EvidenceList title="Key Contributions" items={summary.keyContributions} />
              <EvidenceList title="Claims and Facts" items={summary.claims} />
              <EvidenceList title="Methodology" items={summary.methodology} />
              <EvidenceList title="Datasets and Metrics" items={summary.datasetsAndMetrics} />
              <EvidenceList title="Limitations" items={summary.limitations} />
              <EvidenceList title="Future Work" items={summary.futureWork} />
            </div>
          )}

          <section className="panel compare">
            <h2>Compare Two Papers</h2>
            <form onSubmit={compare}>
              {[0, 1].map((index) => (
                <select key={index} value={compareIds[index]} onChange={(event) => {
                  const next = [...compareIds];
                  next[index] = event.target.value;
                  setCompareIds(next);
                }}>
                  <option value="">Paper {index + 1}</option>
                  {papers.map((paper) => (
                    <option key={paper.id} value={paper.id}>{paper.title}</option>
                  ))}
                </select>
              ))}
              <button disabled={!compareIds[0] || !compareIds[1] || busy === "compare"}>
                {busy === "compare" ? <Loader2 className="spin" size={18} /> : <GitCompareArrows size={18} />}
                Compare
              </button>
            </form>
            {comparison && (
              <div className="comparison">
                <div>
                  <h3>{comparison.papers?.[0]?.title}</h3>
                  <EvidenceStack items={comparison.paperA?.contributions} />
                </div>
                <div>
                  <h3>{comparison.papers?.[1]?.title}</h3>
                  <EvidenceStack items={comparison.paperB?.contributions} />
                </div>
              </div>
            )}
          </section>
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
