# AGT-SS AWS Infrastructure & Deployment

## Architecture overview

```
Internet
   │
   ▼
Application Load Balancer  (public subnets — 2 AZs)
   │  HTTPS/443 → HTTP/80 redirect
   ▼
ECS Fargate Service        (private subnets — 2 AZs)
   │  agt_ss_api FastAPI + agt_ss LangGraph agents
   │  ThreadPoolExecutor for async workflow runs
   ▼
Aurora Serverless v2       (isolated DB subnets — 2 AZs)
   PostgreSQL 15
   workflow_checkpoints | dead_letters | approved_suppliers
   │
   ├── Secrets Manager  (API keys, Anthropic key, DB creds, SAP, DocuSign)
   ├── CloudWatch Logs  (/agt-ss/{env}/api)
   └── CloudWatch Alarms → SNS → Email
```

## Stack deployment order

| # | Stack file | Name pattern | Creates |
|---|------------|--------------|---------|
| 1 | 01-network.yaml | agt-ss-network-{env} | VPC, subnets, NAT, security groups |
| 2 | 02-database.yaml | agt-ss-database-{env} | Aurora Serverless v2 cluster |
| 3 | 03-secrets.yaml | agt-ss-secrets-{env} | Secrets Manager entries |
| 4 | 04-iam-ecr.yaml | agt-ss-iam-ecr-{env} | ECR repo, IAM roles, GitHub OIDC |
| 5 | 05-compute.yaml | agt-ss-compute-{env} | ECS cluster, ALB, task def, service |

## Prerequisites

```bash
# AWS CLI v2
aws --version

# Docker
docker --version

# jq (for deploy script JSON manipulation)
brew install jq   # macOS
apt install jq    # Ubuntu

# AWS credentials with AdministratorAccess (or scoped policy)
aws configure
# or: export AWS_PROFILE=cocom-deploy
```

## First-time full deploy

```bash
# Clone the repo
git clone https://github.com/cocomgroup/agt-ss.git
cd agt-ss

# Make scripts executable
chmod +x infra/scripts/deploy.sh

# Set your AWS account ID (auto-detected if omitted)
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1

# Optional: ACM certificate for HTTPS
export ACM_CERT_ARN=arn:aws:acm:us-east-1:123456789012:certificate/abc-123

# Optional: email for CloudWatch alarm notifications
export ALERT_EMAIL=ops@cocomgroup.com

# Deploy everything (infra + build + push + ECS update)
./infra/scripts/deploy.sh production all
```

This single command:
1. Deploys all 5 CloudFormation stacks in dependency order
2. Builds the Docker image from source
3. Pushes to ECR
4. Updates the ECS task definition and service
5. Waits for service stability
6. Prints the ALB DNS name

## Populate secrets after first deploy

Secrets are created with `REPLACE_ME` placeholder values. Populate them:

```bash
./infra/scripts/deploy.sh production secrets
```

This prompts for:
- Anthropic API key (`sk-ant-...`)
- AGT-SS API key (for `X-API-Key` header)

For SAP and DocuSign, update via CLI:
```bash
aws secretsmanager put-secret-value \
  --secret-id agt-ss/production/sap/credentials \
  --secret-string '{
    "SAP_HOST": "your-sap-host.cocomgroup.com",
    "SAP_CLIENT": "100",
    "SAP_USERNAME": "RFC_USER",
    "SAP_PASSWORD": "your-password",
    "SAP_SYSTEM_ID": "PRD"
  }'
```

## Bootstrap the database

After first deploy, run the SQL bootstrap to create tables and the app user:

```bash
# Option A: via AWS Systems Manager Session Manager (no bastion needed)
aws ecs execute-command \
  --cluster agt-ss-production \
  --task <task-arn> \
  --container api \
  --interactive \
  --command "/bin/bash"

# Inside container:
# psql -h $DB_HOST -U agt_ss_admin -d agt_ss -f /dev/stdin < infra/scripts/bootstrap.sql

# Option B: from a bastion host in the VPC
psql -h <aurora-endpoint> -U agt_ss_admin -d agt_ss \
  -f infra/scripts/bootstrap.sql
```

## Individual commands

```bash
# Deploy CloudFormation stacks only (no Docker build)
./infra/scripts/deploy.sh production infra

# Build and push image only
IMAGE_TAG=v1.2.3 ./infra/scripts/deploy.sh production build

# Update ECS to a specific image tag
IMAGE_TAG=v1.2.3 ./infra/scripts/deploy.sh production service

# Check status of all stacks and ECS health
./infra/scripts/deploy.sh production status

# Populate Secrets Manager interactively
./infra/scripts/deploy.sh production secrets

# Tear down non-production environment
./infra/scripts/deploy.sh development destroy
```

## GitHub Actions CI/CD

The pipeline in `.github/workflows/deploy.yml` runs automatically:

| Trigger | Action |
|---------|--------|
| PR to `main` or `staging` | Run tests only |
| Push to `staging` | Build → push → deploy to staging |
| Push to `main` or `v*` tag | Build → push → deploy to production |

### Setup steps

1. Add GitHub repository secrets:
   - `AWS_ACCOUNT_ID` — your AWS account number

2. The OIDC trust is created by `04-iam-ecr.yaml` using `GitHubOrg` and `GitHubRepo` parameters.
   No static AWS credentials are stored in GitHub.

3. Create GitHub Environments (`staging`, `production`) and add environment-level protection rules for production (required reviewers, etc.).

## Environment variables reference

All variables injected into the ECS task at runtime:

| Variable | Source | Description |
|----------|--------|-------------|
| `ENVIRONMENT` | CloudFormation | `development \| staging \| production` |
| `AWS_REGION` | CloudFormation | e.g. `us-east-1` |
| `ANTHROPIC_API_KEY` | Secrets Manager | Claude API key |
| `API_KEYS` | Secrets Manager | JSON array of valid client API keys |
| `DB_HOST` | CloudFormation (SSM) | Aurora writer endpoint |
| `DB_PORT` | CloudFormation | `5432` |
| `DB_NAME` | CloudFormation | `agt_ss` |
| `DB_USER` | CloudFormation | `agt_ss_app` |
| `DB_PASSWORD` | Secrets Manager | Aurora app user password |
| `RUN_WORKFLOWS_ASYNC` | CloudFormation | `true` in production |
| `WORKFLOW_TIMEOUT_SECONDS` | CloudFormation | `1800` (30 min) |
| `LOG_LEVEL` | CloudFormation | `info` in production |
| `DISABLE_AUTH` | CloudFormation | Always `false` in production |

## Monitoring

### CloudWatch Log Insights queries

```
# All DEAD_LETTER workflows in the last 24h
fields @timestamp, message
| filter message like /DEAD_LETTER/
| sort @timestamp desc
| limit 20

# Slow API requests (> 5s)
fields @timestamp, message
| filter message like /latency_ms/
| parse message "latency_ms=* " as latency
| filter latency > 5000
| sort latency desc

# Checkpoint gate requests by hour
fields @timestamp
| filter message like /checkpoint/
| stats count() by bin(1h)
```

### CloudWatch alarms created automatically

| Alarm | Threshold | Action |
|-------|-----------|--------|
| Unhealthy ECS hosts | > 0 for 2 min | SNS → email |
| ALB P99 latency | > 5s for 15 min | SNS → email |
| HTTP 5xx error rate | > 1% for 10 min | SNS → email |
| ECS CPU utilisation | > 85% for 15 min | SNS → email |
| Dead-letter workflows | > 0 in 5 min | SNS → email |

## Costs (estimated, us-east-1)

| Component | Sizing | Est. monthly |
|-----------|--------|-------------|
| ECS Fargate (2 tasks × 1vCPU/3GB) | ~730 hrs | ~$60 |
| Aurora Serverless v2 (0.5–4 ACU) | light use | $30–$120 |
| ALB | 1 LCU baseline | ~$20 |
| NAT Gateway (production, 2× AZ) | 1 GB/day | ~$65 |
| Secrets Manager (5 secrets) | — | ~$2 |
| CloudWatch Logs (90-day retention) | ~5 GB/mo | ~$3 |
| ECR (10 image tags) | ~2 GB | ~$0.20 |
| **Total (production, light use)** | | **~$180–$270/mo** |

Development environment with 1 NAT, 1 ECS task, Aurora paused: ~$40–$70/mo.

## Security notes

- ECS tasks run in private subnets with no public IP
- ALB is the only internet-facing component
- Aurora is in isolated DB subnets with no route to internet
- All secrets sourced from Secrets Manager at task start — never baked into images
- Container runs as non-root user (`appuser`)
- ECR scan-on-push enabled — Trivy runs in CI for additional coverage
- OIDC authentication for GitHub Actions — no static AWS credentials in CI
- Production Aurora has deletion protection and `DeletionPolicy: Retain`
