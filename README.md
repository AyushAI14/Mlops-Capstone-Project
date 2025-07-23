
---

# 🧠 Capstone ML Project - MLOps Pipeline

This project demonstrates a complete Machine Learning lifecycle built with an MLOps-first mindset. From structured project scaffolding to version-controlled pipelines, experiment tracking, containerization, and CI/CD — this repo showcases how real-world ML projects are built and deployed.

---

## 🚀 Tech Stack

- **Language:** Python 3.10
- **ML Tools:** Scikit-learn, Pandas, MLflow
- **MLOps:** DVC, FastAPI, Docker, GitHub Actions
- **Cloud:** AWS (S3, ECR), Dagshub

---

## 📁 Project Setup

1. **Clone the Repository**  
   ```bash
   git clone <repo_url> && cd <repo>
2. **Create Virtual Environment**

   ```bash
    python3 -m venv env
    source env/bin/activate
   ```

3. **Project Structure via Cookiecutter**

   ```bash
   pip install cookiecutter
   cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
   ```

4. **Post-Creation Cleanup**

   * Rename `src.models` → `src.model`
   * `git add . && git commit -m "Initial structure" && git push`

---

## 📊 Experiment Tracking with MLflow + Dagshub

1. **Connect Repository to Dagshub:**

   * Go to [dagshub.com](https://dagshub.com/dashboard)
   * Create → New Repo → Connect via GitHub

2. **Install Required Packages**

   ```bash
   pip install dagshub mlflow
   ```

3. **Use MLflow tracking code from Dagshub UI**

---

## 🔁 DVC Pipeline Setup

1. **Initialize DVC**

   ```bash
   dvc init
   mkdir local_s3
   dvc remote add -d mylocal local_s3
   ```

2. **Add Pipeline Files in `src/`**

   * `logger.py`, `data_ingestion.py`, `data_preprocessing.py`, `feature_engineering.py`, `model_building.py`, `model_evaluation.py`, `register_model.py`

3. **Create Config Files**

   * `params.yaml`
   * `dvc.yaml` (define stages till model evaluation)

4. **Run and Track Pipeline**

   ```bash
   dvc repro
   dvc status
   git add . && git commit -m "DVC pipeline added" && git push
   ```

---

## ☁️ AWS S3 as DVC Remote

1. **Setup AWS Credentials**

   * Create IAM user with S3 access
   * `aws configure` with Access Key and Secret

2. **Install Dependencies**

   ```bash
   pip install dvc[s3] awscli
   ```

3. **Add S3 Remote**

   ```bash
   dvc remote add -d myremote s3://<your-bucket-name>
   ```

---

## ⚙️ Model Deployment via FastAPI

1. **Create a new directory:** `fastapi_app/`

2. **Install & Run App**

   ```bash
   pip install fastapi uvicorn
   uvicorn main:app --reload
   ```

3. **Push Data to S3**

   ```bash
   dvc push
   ```

---

## 📦 Requirements & CI Setup

1. **Generate Requirements**

   ```bash
   pip freeze > requirements.txt
   ```

2. **Add GitHub Actions Workflow**

   * Path: `.github/workflows/ci.yaml`

3. **Add Dagshub Auth Token**

   * Generate from Dagshub → Your Repo → Settings → Tokens
   * Add as GitHub Secret: `CAPSTONE_TEST`
   * Use in `ci.yaml` workflow

---

## 🐳 Dockerization

1. **Prepare Docker Setup**

   ```bash
   pip install pipreqs
   cd fastapi_app
   pipreqs . --force
   ```

2. **Dockerfile & Build**

   ```bash
   docker build -t capstone-app:latest .
   ```

3. **Run Docker Image**

   ```bash
   docker run -p 8888:5000 -e CAPSTONE_TEST=<your_token> capstone-app:latest
   ```

4. **Optional: Push to DockerHub**

   ```bash
   docker tag capstone-app youruser/capstone-app:latest
   docker push youruser/capstone-app:latest
   ```

---

## 🛠️ AWS ECR & CI/CD Integration

1. **IAM Permissions**

   * Assign `AmazonEC2ContainerRegistryFullAccess` to IAM user

2. **Required Secrets**

   * `AWS_ACCESS_KEY_ID`
   * `AWS_SECRET_ACCESS_KEY`
   * `AWS_REGION`
   * `ECR_REPOSITORY` (e.g., `capstone-proj`)
   * `AWS_ACCOUNT_ID`

3. **CI/CD via GitHub Actions**

   * Automatically build and push Docker image to ECR

---

## ✅ Summary

| Component        | Status        |
| ---------------- | ------------- |
| Project Scaffold | ✅             |
| MLflow Tracking  | ✅ via Dagshub |
| DVC Pipeline     | ✅             |
| AWS S3 Storage   | ✅             |
| FastAPI App      | ✅             |
| Docker Setup     | ✅             |
| CI/CD Workflow   | ✅             |

---

## 👨‍💻 Author

**Ayush**
Machine Learning Engineer | MLOps Enthusiast
Feel free to connect: [GitHub](https://github.com/AyushAI14) • [LinkedIn](https://www.linkedin.com/in/ayush-vishwakarma-a2450a28b/)

---

