# Cloud Module

## Overview

The `cloud` module centralizes all AWS-related functionality used throughout the Financial Machine Learning project.

Rather than embedding AWS logic inside machine learning scripts, cloud operations are isolated into reusable components.

---

# Responsibilities

- AWS configuration
- Amazon S3 communication
- Dataset uploads
- Dataset downloads
- Intelligent synchronization
- Manifest management
- Future SageMaker integration

---

# Architecture

```
Local Project

↓

Cloud Module

↓

Amazon S3

↓

SageMaker
```

---

# File Structure

```
cloud/

config.py

s3_client.py

uploader.py

downloader.py

sync.py

manifest.py

utils.py
```

---

# Synchronization Engine

The synchronization engine performs the following steps.

1. Scan local datasets

2. Calculate MD5 hash

3. Compare against manifest

4. Upload changed files only

5. Update manifest

This minimizes unnecessary uploads and keeps Amazon S3 synchronized with the local project.

---

# Current Status

- AWS CLI configured
- Amazon S3 connected
- Automated uploads implemented
- Incremental synchronization verified
- Manifest tracking operational

---

# Future Enhancements

- SageMaker integration
- Model artifact synchronization
- Automatic prediction uploads
- CloudDataManager abstraction
- Scheduled synchronization

                    Financial ML Platform

                  Yahoo Finance API
                          │
                          ▼
                Local Data Collection
                          │
                          ▼
                Feature Engineering
                          │
                          ▼
                Model Input Datasets
                          │
                          ▼
             Intelligent Cloud Sync Engine
                          │
                          ▼
                  Amazon S3 Data Lake
                  ┌─────────┴─────────┐
                  ▼                   ▼
         Local Development    SageMaker Studio
                  │                   │
                  └─────────┬─────────┘
                            ▼
                     XGBoost Training
                            │
                            ▼
                    Trained ML Models
                            │
                            ▼
                      Amazon S3 Storage
