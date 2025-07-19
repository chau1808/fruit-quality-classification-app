import React, { useState, useRef } from "react";
import Webcam from "react-webcam";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");
  const webcamRef = useRef(null);

  const capture = () => {
    const screenshot = webcamRef.current.getScreenshot();
    fetch(screenshot)
      .then(res => res.blob())
      .then(blob => {
        setImage(blob);
      });
  };

  const handleUpload = (e) => {
    setImage(e.target.files[0]);
  };

  const handlePredict = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("image", image);

    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data.result);
  };

  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      <h1>Fruit Quality Classifier 🍎🍌🍊</h1>

      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={300}
      />
      <br />
      <button onClick={capture}>📸 Chụp ảnh</button>
      <br /><br />

      <input type="file" accept="image/*" onChange={handleUpload} />
      <br /><br />

      <button onClick={handlePredict}>🔍 Dự đoán</button>

      {result && (
        <h2 style={{ marginTop: "1rem" }}>Kết quả: <span>{result}</span></h2>
      )}
    </div>
  );
}

export default App;
