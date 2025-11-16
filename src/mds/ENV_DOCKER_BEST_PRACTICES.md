# Environment Variables Best Practices for Docker

## Why This Matters

Passing environment variables to Docker containers requires careful consideration for:
- **Security**: Prevents secrets from being exposed in containers
- **Configuration Management**: Makes it easy to manage different environments
- **Maintainability**: Keeps configuration separate from code

## Current Implementation

### ✅ What We're Doing Now

**Using `env_file` directive** (Recommended for Docker Compose)

```yaml
services:
  voice-assistant:
    env_file:
      - .env  # Loads variables from .env file
    environment:
      - PYTHONPATH=/app
      # Explicit overrides
```

### Benefits:
1. **Variables are injected, not files**: The `.env` file is read once at container startup
2. **No file mounted**: No risk of accidentally exposing secrets via mounted volumes
3. **Clean separation**: Environment variables are environment variables, not files in the container
4. **Override capability**: Can still override specific vars in the `environment` section

## Comparison: Mounted Files vs `env_file`

### ❌ Mounting Files (Not Recommended)
```yaml
volumes:
  - ./.env:/app/.env  # DON'T DO THIS
```
**Problems:**
- File is physically mounted, visible in container
- Can be read/written by code inside container
- If compromised, `.env` is accessible
- Adds unnecessary files to the container filesystem

### ✅ Using `env_file` (Recommended)
```yaml
env_file:
  - .env
```
**Benefits:**
- Variables are loaded into environment only
- `.env` file not accessible inside container
- Variables can be overridden per-environment
- No file management overhead

## Best Practices by Environment

### 1. Development (docker-compose.yml)

**Option A: `env_file` (Current - Recommended)**
```yaml
env_file:
  - .env
```

**Option B: Explicit Variables (Alternative)**
```yaml
environment:
  - MOCK_LLM=true
  - MOCK_TTS=true
  - LLM_TEMPERATURE=0.7
```

### 2. Production (docker-compose.prod.yml)

**Option A: `env_file` (Current - Good for simple deployments)**
```yaml
env_file:
  - .env.prod  # Separate .env for production
```

**Option B: Docker Secrets (Best for sensitive data)**
```yaml
secrets:
  - db_password
  - api_key

services:
  voice-assistant:
    secrets:
      - source: db_password
        target: DB_PASSWORD
```

**Option C: External Secrets Management (Enterprise)**
- AWS Secrets Manager
- HashiCorp Vault
- Kubernetes Secrets
- Docker Swarm Secrets

## Security Best Practices

### 1. Never Commit `.env` to Git
```bash
# In .gitignore
.env
.env.*
!.env.example
```

### 2. Use Separate Environment Files
```
.env                # Local development
.env.development    # Development environment
.env.staging       # Staging environment
.env.production    # Production environment
```

### 3. Rotate Secrets Regularly
- Change passwords/keys periodically
- Use CI/CD for automatic rotation
- Audit access logs

### 4. Limit Scope of Variables
- Only expose necessary variables
- Use least privilege principle
- Don't expose sensitive data if not needed

## Configuration Hierarchy

Docker Compose processes environment variables in this order (last wins):

1. **Dockerfile ENV defaults**
2. **Compose file `environment` section**
3. **`.env` file**
4. **`env_file` directive**
5. **Explicit `environment` entries** (highest priority)

```yaml
services:
  app:
    environment:
      - DEBUG=false        # 1. Dockerfile default
      - DEBUG=true         # 2. Explicit override - THIS WINS
    env_file:
      - .env              # 3. Loads from .env
```

## Advanced: Different Environments

### Create Multiple .env Files

`.env.development`:
```bash
MOCK_LLM=true
MOCK_TTS=true
DEBUG=true
```

`.env.production`:
```bash
MOCK_LLM=false
MOCK_TTS=false
DEBUG=false
DB_PASSWORD=secure_password_here
```

### Use with Compose
```bash
# Development
docker-compose --env-file .env.development up

# Production
docker-compose --env-file .env.production up
```

### In docker-compose.prod.yml
```yaml
env_file:
  - .env.production
```

## Troubleshooting

### Verify Environment Variables

```bash
# Inside container
docker exec voice-assistant env | grep LLM

# Or start container with shell
docker-compose run voice-assistant sh
env
```

### Debug Loading

```bash
# See what Compose sees
docker-compose config
```

### Common Issues

1. **Variables not loading**: Check `env_file` path is correct
2. **Variables overridden**: Check `environment` section
3. **File not found**: Ensure `.env` exists in project root
4. **Wrong values**: Check variable precedence

## Migration Guide

If you're currently mounting `.env` files:

### Before (Not Recommended)
```yaml
volumes:
  - ./.env:/app/.env
```

### After (Recommended)
```yaml
env_file:
  - .env
```

**Benefits:**
- Better security (no file mounted)
- Cleaner container filesystem
- Follows Docker best practices
- Easier to override per-environment

## References

- [Docker Compose Environment Variables](https://docs.docker.com/compose/environment-variables/)
- [Docker Secrets](https://docs.docker.com/engine/swarm/secrets/)
- [12 Factor App - Config](https://12factor.net/config)
- [OWASP Docker Security](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

## Summary

✅ **DO:**
- Use `env_file` directive
- Keep `.env` out of Git
- Use separate files per environment
- Rotate secrets regularly
- Use Docker secrets for sensitive data in production

❌ **DON'T:**
- Mount `.env` files as volumes
- Commit secrets to version control
- Use same `.env` for all environments
- Hardcode values in Dockerfiles
- Expose more variables than necessary

