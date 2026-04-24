variable "aws_region" {
  description = "The AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "Your AWS Account ID for ECR registry reference."
  type        = string
}

variable "image_version" {
  description = "The tag of the Docker image to deploy."
  type        = string
  default     = "v1.0.0"
}

variable "instance_type" {
  description = "EC2 instance type for the backend (Strategy 1: Single Instance)"
  type        = string
  default     = "t3.micro"
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for model storage (Strategy 3: S3 only for storage)"
  type        = string
  default     = "nextrade-model-storage"
}
