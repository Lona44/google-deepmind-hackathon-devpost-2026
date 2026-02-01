# Cloud Run Services for G1 Platform

# Web Application
resource "google_cloud_run_v2_service" "web" {
  name     = "g1-web"
  location = var.region

  template {
    service_account = google_service_account.web_app.email

    containers {
      image = "gcr.io/${var.project_id}/g1-web:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "RESULTS_BUCKET"
        value = google_storage_bucket.results.name
      }

      env {
        name  = "VIEWER_URL"
        value = "https://g1-viewer-${var.project_id}.web.app"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access to web app
resource "google_cloud_run_v2_service_iam_member" "web_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.web.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Run Job for Workers
resource "google_cloud_run_v2_job" "worker" {
  name     = "g1-worker"
  location = var.region

  template {
    template {
      service_account = google_service_account.worker.email
      timeout         = "3600s" # 1 hour max

      containers {
        image = "gcr.io/${var.project_id}/g1-worker:latest"

        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }

        env {
          name  = "RESULTS_BUCKET"
          value = google_storage_bucket.results.name
        }

        resources {
          limits = {
            cpu    = "4"
            memory = "8Gi"
          }
        }
      }

      max_retries = 1
    }
  }

  depends_on = [google_project_service.apis]
}

# Outputs
output "web_url" {
  value = google_cloud_run_v2_service.web.uri
}

output "worker_job_name" {
  value = google_cloud_run_v2_job.worker.name
}
