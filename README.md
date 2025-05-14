# StyleSync

**StyleSync** is an intelligent fashion search platform that enables users to discover fashion items through image and text queries. Leveraging deep learning and vector similarity search, StyleSync provides fast and accurate results for both image and text-based searches.

## Features

- **Multimodal Search**: Search for fashion items using uploaded images, real-time webcam captures, or text descriptions.
- **Deep Learning Powered**: Utilizes a CLIP-based feature extractor for robust image and text understanding.
- **Fast Vector Search**: Integrates with Pinecone for scalable and efficient vector similarity retrieval.
- **Efficient Storage**: Uses MongoDB with GridFS to store and serve large-scale fashion item images.
- **User-Friendly Interface**: Built with Flask to provide a clean and intuitive web application.

##  Tech Stack

###  Backend
- Python
- Flask

###  Deep Learning
- OpenAI CLIP (via Hugging Face Transformers or OpenCLIP)

###  Vector Search
- Pinecone

###  Image Storage
- MongoDB + GridFS

### Frontend
- HTML
- CSS
