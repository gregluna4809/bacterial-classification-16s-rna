import { useMemo, useState } from "react";

type PredictionItem = {
  genus: string;
  probability: number;
};

type PredictResponse = {
  predicted_genus: string;
  confidence: number;
  top_5: PredictionItem[];
  sequence_length: number;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const EXAMPLE_FASTA = `>Example_16S_sequence
AGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGAACG
GGAAAGGCCCTTCGGGGTACACGAGCGGCGAACGGGTGAGTAACACGTGGGTAACCTGCCC
TTAAGACTGGGATAACTCCGGGAAACCGGGGCTAATACCGGATAACATTTTGAACCGCATG
GTTCGAAATTGAAAGGCGGCTTCGGCTGTCACTTATAGATGGACCCGCGGCGCATTAGCTA
GTTGGTGAGGTAACGGCTCACCAAGGCGACGATGCGTAGCCGACCTGAGAGGGTGATCGGC
CACACTGGGACTGAGACACGGCCCAGACTCCTACGGGAGGCAGCAGTGGGGAATATTGCAC
AATGGGCGCAAGCCTGATGCAGCGACGCCGCGTGAGGGATGACGGCCTTCGGGTTGTAAAC
CTCTTTCAGCAGGGAAGAAGCGAAAGTGACGGTACCTGCAGAAGAAGCACCGGCTAACTAC
GTGCCAGCAGCCGCGGTAATACGTAGGGTGCGAGCGTTGTCCGGAATTATTGGGCGTAAAG
AGCTCGTAGGCGGTTTGTCGCGTCGGTTGTGAAAGCCCGGGGCTTAACCCCGGGTCTGCAG`;

function cleanFastaInput(value: string): string {
  return value
    .split(/\r?\n/)
    .filter((line) => !line.trim().startsWith(">"))
    .join("")
    .replace(/\s+/g, "")
    .toUpperCase();
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function App() {
  const [sequenceInput, setSequenceInput] = useState("");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const cleanedSequence = useMemo(() => cleanFastaInput(sequenceInput), [sequenceInput]);

  async function handlePredict() {
    setError(null);
    setPrediction(null);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: cleanedSequence }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Prediction request failed.");
      }

      setPrediction(payload as PredictResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  }

  function handleClear() {
    setSequenceInput("");
    setPrediction(null);
    setError(null);
  }

  function handleLoadExample() {
    setSequenceInput(EXAMPLE_FASTA);
    setPrediction(null);
    setError(null);
  }

  return (
    <main className="app-shell">
      <section className="header-band">
        <div>
          <p className="eyebrow">Capstone inference demo</p>
          <h1>16S rRNA Bacterial Genus Classifier</h1>
        </div>
        <div className="status-pill">Local API</div>
      </section>

      <section className="dashboard-grid">
        <div className="workspace-panel">
          <div className="panel-header">
            <div>
              <h2>Sequence Input</h2>
              <p>Paste a raw 16S sequence or FASTA-formatted text.</p>
            </div>
            <span>{cleanedSequence.length.toLocaleString()} bp</span>
          </div>

          <textarea
            value={sequenceInput}
            onChange={(event) => setSequenceInput(event.target.value)}
            spellCheck={false}
            placeholder="Paste sequence or FASTA here..."
          />

          <div className="button-row">
            <button className="primary" disabled={isLoading} onClick={handlePredict}>
              {isLoading ? "Predicting..." : "Predict"}
            </button>
            <button onClick={handleLoadExample}>Load Example</button>
            <button onClick={handleClear}>Clear</button>
          </div>

          {error && <div className="error-box">{error}</div>}
        </div>

        <aside className="summary-panel">
          <h2>Model Summary</h2>
          <div className="metric-list">
            <div>
              <span>Accuracy</span>
              <strong>98.5%+</strong>
            </div>
            <div>
              <span>Coverage</span>
              <strong>Top 100 genera</strong>
            </div>
            <div>
              <span>Pipeline</span>
              <strong>XGBoost + TruncatedSVD + SMOTE</strong>
            </div>
          </div>
        </aside>
      </section>

      <section className="results-panel">
        <div className="panel-header">
          <div>
            <h2>Prediction Results</h2>
            <p>Confidence scores are returned by the saved classifier.</p>
          </div>
        </div>

        {prediction ? (
          <>
            <div className="result-cards">
              <div>
                <span>Predicted genus</span>
                <strong>{prediction.predicted_genus}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>{formatPercent(prediction.confidence)}</strong>
              </div>
              <div>
                <span>Sequence length</span>
                <strong>{prediction.sequence_length.toLocaleString()} bp</strong>
              </div>
            </div>

            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Genus</th>
                  <th>Probability</th>
                </tr>
              </thead>
              <tbody>
                {prediction.top_5.map((item, index) => (
                  <tr key={`${item.genus}-${index}`}>
                    <td>{index + 1}</td>
                    <td>{item.genus}</td>
                    <td>{formatPercent(item.probability)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <div className="empty-state">Run a prediction to view genus probabilities.</div>
        )}
      </section>
    </main>
  );
}

export default App;
