import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { motion, AnimatePresence } from "framer-motion";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";

function useDarkMode() {
  const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("theme-dark");
    return saved ? saved === "1" : !!prefersDark;
  });
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("theme-dark", dark ? "1" : "0");
  }, [dark]);
  return [dark, setDark];
}

function ConfidenceTable({ scores }) {
  const rows = useMemo(() => {
    try {
      if (!scores) return [];
      return Object.entries(scores)
        .map(([k, v]) => [k, Number(v)])
        .sort((a, b) => b[1] - a[1]);
    } catch {
      return [];
    }
  }, [scores]);

  if (!rows.length) return null;

  return (
    <div className="card section">
      <h4 className="section-title">🔎 Confidence chi tiết</h4>
      <div className="conf-grid conf-header">
        <div>Lớp</div>
        <div>Độ tin cậy</div>
      </div>
      {rows.map(([label, pct]) => (
        <div className="conf-grid" key={label}>
          <div>{label}</div>
          <div>{pct.toFixed(2)}%</div>
        </div>
      ))}
    </div>
  );
}

function MetaPanel({ meta }) {
  if (!meta) return null;
  return (
    <div className="card section">
      <h4 className="section-title">ℹ️ Thông tin suy luận</h4>
      <div className="meta-row">
        <span className="meta-k">⏱️ Inference</span>
        <span className="meta-v">{meta.inferenceMs ?? "—"} ms</span>
      </div>
      <div className="meta-row">
        <span className="meta-k">🧠 Model</span>
        <span className="meta-v">
          {meta.model?.arch ?? "—"} <span className="pill">v{meta.model?.version ?? "—"}</span>
        </span>
      </div>
      <div className="meta-row">
        <span className="meta-k">📷 Ảnh</span>
        <span className="meta-v">
          {meta.input?.original_size?.w ?? "—"}×{meta.input?.original_size?.h ?? "—"} → <em>224×224</em>
        </span>
      </div>
      {meta.threshold && (
        <div className={`threshold ${meta.threshold.met ? "ok" : "warn"}`}>
          Ngưỡng {meta.threshold.value_pct}% · {meta.threshold.met ? "Đạt" : "Chưa đạt"}
          {!meta.threshold.met && meta.threshold.note ? ` — ${meta.threshold.note}` : ""}
        </div>
      )}
      {meta.topk?.length > 1 && (
        <div className="conf-table">
          <div className="conf-grid conf-header">
            <div>🏅 Top-K</div>
            <div>Độ tin cậy</div>
          </div>
          {meta.topk.map(({ class: label, pct }) => (
            <div className="conf-grid" key={label}>
              <div>{label}</div>
              <div>{Number(pct).toFixed(2)}%</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ProgressBar({ show }) {
  return (
    <div className={`progress-wrap ${show ? "show" : ""}`}>
      <div className="progress-bar" />
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="card section skeleton">
      <div className="sk-title" />
      <div className="sk-line" />
      <div className="sk-line wide" />
      <div className="sk-line" />
    </div>
  );
}

function App() {
  const [dark, setDark] = useDarkMode();

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);         // ảnh gốc (local)
  const [procPreview, setProcPreview] = useState(null); // ảnh 224x224 từ backend
  const [result, setResult] = useState("");
  const [scores, setScores] = useState(null);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [toast, setToast] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const showToast = (msg, t = 2500) => {
    setToast(msg);
    setTimeout(() => setToast(null), t);
  };

  const onPickFile = () => fileInputRef.current?.click();

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showToast("File phải là ảnh (jpg/png/webp...)");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setProcPreview(null); // reset preview 224 khi chọn ảnh mới
    setResult("");
    setScores(null);
    setMeta(null);
  };

  const handleFileChange = (e) => handleFile(e.target.files?.[0]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const handleUpload = async () => {
    if (!image) {
      showToast("Vui lòng chọn hoặc kéo-thả một ảnh!");
      return;
    }
    setLoading(true);
    setScores(null);
    setMeta(null);
    setProcPreview(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const data = await res.json();

      if (res.ok && (data.success || data.prediction)) {
        const className = data.prediction?.class || data.class || "unknown";
        const rawPct = data.prediction?.pct ?? data.confidence ?? 0;
        const pct = Number(rawPct <= 1 ? rawPct * 100 : rawPct);

        setResult(`${className.toUpperCase()} (${pct.toFixed(1)}%)`);
        setScores(data.confidence_scores || null);
        setMeta({
          inferenceMs: data.timings?.inference_ms || data.inference_time,
          model: data.model || (data.model_version ? { arch: "vgg16", version: data.model_version } : null),
          input: data.input || (data.original_size ? { original_size: data.original_size } : null),
          threshold: data.threshold,
          topk: data.top_k || [],
        });

        // 👉 lấy ảnh 224×224 từ backend (data.preview là data:image/png;base64,...)
        setProcPreview(data.preview || null);
      } else {
        setResult(data.error || "Không nhận dạng được");
      }
    } catch (err) {
      console.error(err);
      setResult("Đã xảy ra lỗi khi kết nối server");
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraOn(true);
        showToast("Đang sử dụng camera sau (nếu có).");
      }
    } catch (err) {
      console.error("Camera error:", err);
      showToast("Không thể mở camera. Hãy cấp quyền.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraOn(false);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, 224, 224);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], "camera.jpg", { type: "image/jpeg" });
          if (preview) URL.revokeObjectURL(preview);
          setImage(file);
          setPreview(URL.createObjectURL(file));
          setProcPreview(null);
          setResult("");
          setScores(null);
          setMeta(null);
        }
      },
      "image/jpeg",
      0.95
    );
  };

  return (
    <div className="shell">
      <ProgressBar show={loading} />

      <motion.div
        className="app-container"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <header className="header">
          <div className="brand">
            <span className="logo">🍍</span>
            <h1>Fruit Quality Classifier</h1>
            <span className="beta">VGG16</span>
          </div>
          <div className="actions">
            <button className="ghost" onClick={() => setDark((d) => !d)}>
              {dark ? "🌙" : "☀️"}
            </button>
          </div>
        </header>

        <p className="subtitle">Đánh giá chất lượng trái cây bằng AI · Flask API @ <span className="link">{API_URL}</span></p>

        {/* Upload & DnD */}
        <div
          className={`dnd ${dragOver ? "over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <div className="dnd-inner">
            <div className="dnd-icon">⬆️</div>
            <div className="dnd-title">Kéo-thả ảnh vào đây</div>
            <div className="dnd-sub">hoặc</div>
            <button className="btn" onClick={onPickFile}>Chọn file ảnh</button>
            <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange} />
          </div>
        </div>

        <div className="toolbar">
          {!cameraOn ? (
            <button className="btn secondary" onClick={startCamera}>🎥 Sử dụng camera</button>
          ) : (
            <button className="btn danger" onClick={stopCamera}>⛔ Tắt camera</button>
          )}
          <button className="btn primary" onClick={handleUpload} disabled={loading || !image}>
            {loading ? "🔄 Đang xử lý..." : "🔍 Phân loại"}
          </button>
          {cameraOn && (
            <button className="btn accent" onClick={captureImage}>📸 Chụp ảnh</button>
          )}
        </div>

        <AnimatePresence>
          {cameraOn && (
            <motion.div
              className="camera card"
              initial={{ scale: 0.98, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <video ref={videoRef} autoPlay width="320" height="240" />
              <canvas ref={canvasRef} width="224" height="224" style={{ display: "none" }} />
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid">
          <div className="col">
            {preview ? (
              <div className="card image-card">
                <h3>🖼️ Ảnh đã chọn</h3>
                <img src={preview} alt="Xem trước" />
              </div>
            ) : (
              <SkeletonCard />
            )}

            {/* Ảnh 224x224 từ backend */}
            {procPreview && (
              <div className="card image-card" style={{ marginTop: 12 }}>
                <h3>🧪 Ảnh 224×224 (đưa vào model)</h3>
                <img src={procPreview} alt="Processed 224x224" />
              </div>
            )}

            {result && (
              <div className={`card result-card ${loading ? "loading" : ""}`}>
                <div className="result-line">
                  <span className="badge">Kết quả</span>
                  <span className="result-text">{result}</span>
                </div>
              </div>
            )}
          </div>

          <div className="col">
            {loading ? (
              <>
                <SkeletonCard />
                <SkeletonCard />
              </>
            ) : (
              <>
                <ConfidenceTable scores={scores} />
                <MetaPanel meta={meta} />
              </>
            )}
          </div>
        </div>

        <footer className="footer">
          <div>🌱 VGG16 + Transfer Learning · Xử lý cục bộ</div>
          <div>🛠️ React · Framer Motion · Flask</div>
        </footer>
      </motion.div>

      <AnimatePresence>
        {toast && (
          <motion.div
            className="toast"
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {toast}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
