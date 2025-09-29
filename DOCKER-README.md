# Docker Deployment Guide for Clause_IQ

This guide will help you deploy the Clause_IQ application using Docker on **Mac**, **Windows**, and **Linux**.

## Prerequisites

- Docker Desktop (Mac/Windows) or Docker Engine (Linux)
- Docker Compose v2.0 or higher
- At least 8GB of available RAM
- 10GB of free disk space

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The application will be available at `http://localhost:5000`

### 2. Build and Run with Docker Commands

```bash
# Build the image
docker build -t clause_iq:latest .

# Run the container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/standard_tempate_default:/app/standard_tempate_default \
  --name clause_iq_app \
  clause_iq:latest

# View logs
docker logs -f clause_iq_app

# Stop the container
docker stop clause_iq_app
docker rm clause_iq_app
```

## Platform-Specific Instructions

### Mac (Apple Silicon M1/M2/M3)

If you're using Apple Silicon, the Docker images will work automatically. However, for optimal performance:

```bash
# Build for your platform
docker-compose build --build-arg BUILDPLATFORM=linux/arm64

# Or for Docker command
docker build --platform linux/arm64 -t clause_iq:latest .
```

### Windows

**Using PowerShell:**

```powershell
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

**Using Command Prompt:**

```cmd
docker-compose up -d
docker-compose logs -f
```

**Note for Windows:** Make sure Docker Desktop is running and WSL2 integration is enabled in Docker Desktop settings.

### Linux

```bash
# Install Docker and Docker Compose if not already installed
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Build and run (without sudo if user is in docker group)
docker-compose up -d
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your specific configuration:

```env
FLASK_ENV=production
SECRET_KEY=your-secret-key
GOOGLE_API_KEY=your-api-key
HUGGINGFACE_TOKEN=your-token
```

### Resource Limits

Adjust resource limits in `docker-compose.yml` based on your system:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'      # Adjust CPU limit
      memory: 4G     # Adjust memory limit
```

## Development Mode

To run in development mode with hot-reload:

1. Uncomment the volume mounts in `docker-compose.yml`:

```yaml
volumes:
  - ./app.py:/app/app.py
  - ./helper.py:/app/helper.py
  - ./prompts.py:/app/prompts.py
```

2. Change the CMD in Dockerfile to use Flask development server:

```dockerfile
CMD ["python", "app.py"]
```

3. Update app.py to run on 0.0.0.0:

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, change it in `docker-compose.yml`:

```yaml
ports:
  - "8080:5000"  # Use port 8080 instead
```

### Out of Memory

Increase Docker memory allocation:
- **Mac/Windows:** Docker Desktop → Settings → Resources → Memory
- **Linux:** Edit `/etc/docker/daemon.json`

### Permission Denied (Linux)

Add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Slow Build Times

Use Docker BuildKit for faster builds:

```bash
DOCKER_BUILDKIT=1 docker build -t clause_iq:latest .
```

Or add to `.bashrc`/`.zshrc`:

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Container Health Check Failing

Check container logs:

```bash
docker-compose logs clause_iq
```

Or inspect the health status:

```bash
docker inspect --format='{{json .State.Health}}' clause_iq_app | jq
```

## Useful Commands

```bash
# View running containers
docker-compose ps

# Rebuild after code changes
docker-compose up -d --build

# View container resource usage
docker stats clause_iq_app

# Execute commands inside container
docker-compose exec clause_iq bash

# Clean up everything
docker-compose down -v
docker system prune -a

# Export logs
docker-compose logs > logs.txt
```

## Production Deployment

For production deployment:

1. **Use a reverse proxy** (nginx/traefik) for SSL/TLS
2. **Set proper environment variables** in `.env`
3. **Use Docker secrets** for sensitive data
4. **Enable logging** to external service
5. **Set up monitoring** (Prometheus/Grafana)
6. **Configure backups** for volumes
7. **Use orchestration** (Kubernetes/Docker Swarm) for scaling

