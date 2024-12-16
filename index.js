const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const { v4: uuidv4 } = require('uuid');

// Konfigurasi aplikasi
const app = express();
const port = 8080;

// Middleware
app.use(express.json());

// Konfigurasi Multer untuk upload file
const upload = multer({
  limits: { fileSize: 1000000 },
});

// Inisialisasi Google Cloud Storage
const storage = new Storage();
const bucketName = 'bucket-mlgc-mita';
const modelFileName = 'model.json';

// Inisialisasi Firestore
const firestore = new Firestore();
const predictionsCollection = firestore.collection('predictions');

// Load model dari Cloud Storage
let model;
async function loadModel() {
  try {
    const modelPath = `gs://${bucket-mlgc-mita}/${model.json}`;
    model = await tf.loadGraphModel(modelPath);
    console.log('Model loaded successfully!');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

// Endpoint untuk prediksi
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    // Validasi file
    if (!req.file) {
      return res.status(400).json({ status: 'fail', message: 'No file uploaded' });
    }

    // Konversi buffer ke tensor
    const buffer = req.file.buffer;
    const imageTensor = tf.node.decodeImage(buffer, 3)
      .resizeNearestNeighbor([224, 224])
      .expandDims(0)
      .toFloat()
      .div(255.0);

    // Prediksi
    const prediction = model.predict(imageTensor);
    const predictionValue = prediction.dataSync()[0]; // Nilai prediksi
    const result = predictionValue > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion = result === 'Cancer'
      ? 'Segera periksa ke dokter!'
      : 'Penyakit kanker tidak terdeteksi.';

    // Buat respons
    const id = uuidv4();
    const createdAt = new Date().toISOString();
    const responseData = { id, result, suggestion, createdAt };

    // Simpan ke Firestore
    await predictionsCollection.doc(id).set(responseData);

    // Kirim respons
    res.status(200).json({
      status: 'success',
      message: 'Model is predicted successfully',
      data: responseData,
    });
  } catch (error) {
    console.error(error);
    res.status(400).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
});

// Endpoint untuk cek server
app.get('/', (req, res) => res.send('Backend is running!'));

// Start server
app.listen(port, async () => {
  await loadModel();
  console.log(`Server is running on http://localhost:${port}`);
});
