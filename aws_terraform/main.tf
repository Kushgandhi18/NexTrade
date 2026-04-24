terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------------------------------------------------
# VPC & NETWORK
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_vpc" "nextrade_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "nextrade-vpc"
  }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.nextrade_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.nextrade_vpc.id
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.nextrade_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public_rt.id
}

# ---------------------------------------------------------------------------------------------------------------------
# S3 STORAGE (MODELS & LOGS) - Strategy 3
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_s3_bucket" "model_storage" {
  bucket        = var.s3_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "model_storage_block" {
  bucket = aws_s3_bucket.model_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------------------------------------------------
# EC2 SINGLE INSTANCE (FASTAPI, REDIS, DB, FAISS) - Strategy 1
# ---------------------------------------------------------------------------------------------------------------------
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023*-x86_64"]
  }
}

resource "aws_security_group" "ec2_sg" {
  name        = "nextrade-ec2-sg"
  vpc_id      = aws_vpc.nextrade_vpc.id
  description = "Allow web and SSH traffic"

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "backend" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public_a.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  # Increase root volume for local DB and models
  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  tags = {
    Name = "nextrade-cost-optimized-backend"
  }

  # Strategy 4: Install local Redis and Postgres via Docker
  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y docker python3-pip
              systemctl start docker
              systemctl enable docker
              
              # Install Docker Compose
              mkdir -p /usr/local/lib/docker/cli-plugins/
              curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
              chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
              
              # Set up application directory
              mkdir -p /app/model_store
              EOF
}

# ---------------------------------------------------------------------------------------------------------------------
# IAM ROLES & PROFILES
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_iam_role" "ec2_role" {
  name = "nextrade-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "nextrade-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

resource "aws_iam_policy" "ec2_s3_policy" {
  name        = "nextrade-ec2-s3-policy"
  description = "Allow EC2 to access S3 for models and logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:*"]
        Resource = [
          aws_s3_bucket.model_storage.arn,
          "${aws_s3_bucket.model_storage.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_attachment" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_s3_policy.arn
}
