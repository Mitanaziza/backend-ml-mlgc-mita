const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { Firestore } = require('@google-cloud/firestore');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

// Konfigurasi aplikasi
const app = express();
const port = 8080;

// Middleware
app.use(express.json());

// Konfigurasi Multer untuk upload file
const upload = multer({
  limits: { fileSize: 1000000 },
});


// Inisialisasi Firestore
const firestore = new Firestore({
  projectId: process.env.PROJECT_ID,
  keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS
});
const predictionsCollection = firestore.collection('predictions');

// Load model dari Cloud Storage ke lokal terlebih dahulu
let model;
async function loadModel() {
  try {
    model = await tf.loadGraphModel(process.env.MODEL_URL);
    console.log('Model loaded successfully!');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

// Endpoint untuk prediksi
app.post('/predict', (req, res) => {
  const uploadForm = upload.single('image');
  uploadForm(req, res, async error=> {
    if (error instanceof multer.MulterError) {
      return res.status(413).json({ status: 'fail', message: 'Payload content length greater than maximum allowed: 1000000' });
    }
    try {
      // Validasi file
      if (!req.file) {
        return res.status(400).json({ status: 'fail', message: 'Terjadi kesalahan dalam melakukan prediksi' });
      }
  
      // Konversi buffer ke tensor
      const buffer = req.file.buffer;
      const imageTensor = tf.node.decodeImage(buffer)
        .resizeNearestNeighbor([224, 224])
        .expandDims(0)
        .toFloat()
  
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
      res.status(201).json({
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
  })
});

// Define the /predict/histories endpoint
app.get('/predict/histories', async (req, res) => {
  try {
    // Reference to the Firestore collection where prediction histories are stored
    const historiesCollection = firestore.collection('prediction_histories');

    // Fetch all documents from the collection
    const snapshot = await historiesCollection.get();

    if (snapshot.empty) {
      return res.status(200).json({
        status: 'success',
        data: [],
      });
    }

    // Format the response data
    const histories = snapshot.docs.map(doc => {
      const data = doc.data();
      return {
        id: doc.id,
        history: {
          result: data.result,
          createdAt: data.createdAt,
          suggestion: data.suggestion,
          id: doc.id,
        },
      };
    });

    // Send the response
    res.status(200).json({
      status: 'success',
      data: histories,
    });
  } catch (error) {
    console.error('Error fetching prediction histories:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to fetch prediction histories',
    });

    // Tangani error server dengan status 500
    res.status(500).json({
      status: "fail",
      message: "Failed to retrieve prediction histories"
    });
  }
});



// Endpoint untuk cek server
app.get('/', (req, res) => res.send('Backend is running!'));

// Start server
app.listen(port, '0.0.0.0', async () => {
    await loadModel();
    console.log(`Server is running on http://localhost:${port}`);
  });


