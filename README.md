# Resume NER System - Cloud-Ready Production Environment ☁️🎓

A production-ready AI Resume Information Extraction System built for cloud deployment. This system extracts structured entities (Name, Email, Skills, Experience, etc.) from CVs using a fine-tuned BERT model with LoRA adapters.

## 🌟 Features
- **High-Accuracy NLP**: Fine-tuned `dslim/bert-base-NER` base model utilizing LoRA.
- **Confidence Routing**:
  - High confidence → Auto-accepted
  - Medium confidence → Flagged warning
  - Low confidence → Human review queue
- **Cloud-Native Architecture**: Dockerized frontend & backend, ready for AWS deployment.
- **AWS S3 Integration**: Securely stores uploaded PDF resumes.
- **Security Middleware**: Size limits, payload validation, and API keys.
- **Streamlit Dashboard**: Modern, dark-themed UI for testing and human review queue.

## 📂 Project Structure
```text
.
├── backend/                  # FastAPI backend
│   ├── core/                 # Config & Logging setup
│   ├── middleware/           # Security & validation limits
│   ├── models/               # LoRA adapter & base config
│   ├── services/             # External services (e.g., AWS S3)
│   ├── main.py               # Application entrypoint
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # Streamlit frontend
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docs/                     # Deployment, Architecture, Discussion Notes
├── tests/                    # API & Load Tests
├── .github/workflows/        # CI/CD config
└── docker-compose.yml
```

## 🚀 Quickstart (Docker)

1. Clone the repository and configure `.env`:
```bash
cp .env.example .env
# Edit .env with your AWS credentials if testing S3 upload
```

2. Build and run the containers:
```bash
docker-compose up --build -d
```

3. Access the application:
- **Frontend Dashboard**: [http://localhost:8501](http://localhost:8501)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## ☁️ AWS Deployment
See [`docs/deployment_guide.md`](./docs/deployment_guide.md) for full EC2/ECS and S3 setup instructions.

## 📊 Evaluation Metrics
- **Test F1**: 0.8368
- **Val F1**: 0.8324
- **Skills F1**: 0.7273

## 🔒 Security
- Request payload size limits implemented via custom Middleware.
- API requests can be secured using an `X-API-Key` header.
- AWS S3 bucket configured with IAM roles following the Principle of Least Privilege.
