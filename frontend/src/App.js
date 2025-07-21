import React, { useState, useRef } from 'react';
import './App.css';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult('');
    }
  };

  const handleUpload = async () => {
    if (!image) {
      alert('Vui lòng chọn một ảnh!');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (response.ok && data.success) {
        // Backend trả về format mới với confidence (0-1)
        const confidence = data.confidence || 0;
        const className = data.class || 'unknown';
        
        setResult(`${className.toUpperCase()} (${(confidence * 100).toFixed(1)}%)`);
        
        // Log chi tiết confidence scores nếu có
        if (data.confidence_scores) {
          console.log('Confidence scores:', data.confidence_scores);
        }
      } else {
        setResult(data.error || 'Không nhận dạng được');
      }

    } catch (error) {
      console.error('Lỗi khi gửi ảnh:', error);
      setResult('Đã xảy ra lỗi khi kết nối server');
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraOn(true);
      }
    } catch (err) {
      console.error('Camera error:', err);
      alert('Không thể mở camera. Vui lòng cho phép quyền truy cập camera.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraOn(false);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, 224, 224);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'camera.jpg', { type: 'image/jpeg' });
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult('');
          }
        }, 'image/jpeg');
      }
    }
  };

  return (
    <motion.div className="app-container" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }}>
      <motion.h1 layoutId="title" className="main-title">🍓 Fruit Quality Classifier 🍍</motion.h1>
      <motion.p className="subtitle" initial={{ y: -10, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
        Đánh giá chất lượng trái cây bằng AI 📸
      </motion.p>

      <div className="option-section">
        <motion.div className="option-box" whileHover={{ scale: 1.05 }}>
          <label className="file-label">
            📁 Tải ảnh từ thiết bị
            <input type="file" accept="image/*" onChange={handleFileChange} hidden />
          </label>
        </motion.div>

        <motion.div className="option-box" whileHover={{ scale: 1.05 }}>
          {!cameraOn ? (
            <button className="camera-button" onClick={startCamera}>🎥 Sử dụng camera</button>
          ) : (
            <button className="camera-button stop" onClick={stopCamera}>⏹️ Tắt camera</button>
          )}
        </motion.div>
      </div>

      <div className="button-group">
        <button 
          className="upload-button" 
          onClick={handleUpload}
          disabled={loading || !image}
        >
          {loading ? '🔄 Đang xử lý...' : '🔍 Phân loại'}
        </button>
        
        {cameraOn && (
          <button className="capture-button" onClick={captureImage}>📸 Chụp ảnh</button>
        )}
      </div>

      <AnimatePresence>
        {cameraOn && (
          <motion.div 
            className="camera-section" 
            initial={{ scale: 0.8, opacity: 0 }} 
            animate={{ scale: 1, opacity: 1 }} 
            exit={{ opacity: 0 }} 
            transition={{ duration: 0.4 }}
          >
            <video ref={videoRef} autoPlay width="300" height="225" />
            <canvas ref={canvasRef} width="224" height="224" style={{ display: 'none' }} />
          </motion.div>
        )}

        {preview && (
          <motion.div 
            className="image-preview" 
            initial={{ y: 30, opacity: 0 }} 
            animate={{ y: 0, opacity: 1 }} 
            exit={{ opacity: 0 }} 
            transition={{ duration: 0.4 }}
          >
            <h3>🖼️ Ảnh đã chọn</h3>
            <img src={preview} alt="Xem trước" />
          </motion.div>
        )}

        {result && (
          <motion.div 
            className={`result ${loading ? 'loading' : ''}`}
            initial={{ scale: 0.8, opacity: 0 }} 
            animate={{ scale: 1, opacity: 1 }} 
            transition={{ duration: 0.4 }}
          >
            <h3>📊 Kết quả phân loại:</h3>
            <p>{result}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.footer className="footer" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
        <p>🌱 Ứng dụng sử dụng AI (VGG16 + Transfer Learning) để đánh giá chất lượng giỏ trái cây.</p>
        <p>💡 Hỗ trợ ảnh từ thiết bị hoặc chụp qua camera. Dữ liệu được xử lý cục bộ để bảo mật.</p>
        <p>🛠️ Công nghệ: React, JavaScript, Framer Motion, Flask API.</p>
        <p>🔗 Backend: <span style={{color: '#4CAF50'}}>http://127.0.0.1:5000</span></p>
      </motion.footer>
    </motion.div>
  );
}

export default App;