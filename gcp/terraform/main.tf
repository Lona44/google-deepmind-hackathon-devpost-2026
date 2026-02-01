# G1 Alignment Platform - GCP Infrastructure
# Terraform configuration for Cloud Run, Storage, Firestore, etc.

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "g1-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "pubsub.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# Firestore Database
resource "google_firestore_database" "main" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.apis]
}

# Cloud Storage Bucket for Results
resource "google_storage_bucket" "results" {
  name          = "g1-results-${var.project_id}"
  location      = var.region
  force_destroy = var.environment != "prod"

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90 # Delete after 90 days in non-prod
    }
    action {
      type = "Delete"
    }
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Pub/Sub Topic for Job Queue
resource "google_pubsub_topic" "experiment_jobs" {
  name = "experiment-jobs"

  depends_on = [google_project_service.apis]
}

# Pub/Sub Subscription for Workers
resource "google_pubsub_subscription" "worker_subscription" {
  name  = "experiment-jobs-worker"
  topic = google_pubsub_topic.experiment_jobs.name

  ack_deadline_seconds = 600 # 10 minutes for long-running jobs

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }
}

# Dead Letter Topic
resource "google_pubsub_topic" "dead_letter" {
  name = "experiment-jobs-dead-letter"
}

# Service Account for Web App
resource "google_service_account" "web_app" {
  account_id   = "g1-web-app"
  display_name = "G1 Web Application"
}

# Service Account for Worker
resource "google_service_account" "worker" {
  account_id   = "g1-worker"
  display_name = "G1 Experiment Worker"
}

# IAM: Web App permissions
resource "google_project_iam_member" "web_app_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.web_app.email}"
}

resource "google_project_iam_member" "web_app_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.web_app.email}"
}

resource "google_storage_bucket_iam_member" "web_app_storage" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.web_app.email}"
}

# IAM: Worker permissions
resource "google_project_iam_member" "worker_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_storage_bucket_iam_member" "worker_storage" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_project_iam_member" "worker_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# Outputs
output "results_bucket" {
  value = google_storage_bucket.results.name
}

output "pubsub_topic" {
  value = google_pubsub_topic.experiment_jobs.name
}

output "web_app_service_account" {
  value = google_service_account.web_app.email
}

output "worker_service_account" {
  value = google_service_account.worker.email
}
