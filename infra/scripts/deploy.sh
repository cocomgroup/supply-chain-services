#!/usr/bin/env bash
# =============================================================================
# deploy.sh — AGT-SS full infrastructure provisioning and deployment script
#
# Usage:
#   ./infra/scripts/deploy.sh [ENVIRONMENT] [COMMAND]
#
# Environments:  development | staging | production   (default: development)
# Commands:
#   all           Deploy all stacks + build + push + update ECS  (default)
#   infra         Deploy CloudFormation stacks only
#   build         Build and push Docker image only
#   service       Update ECS service with latest image
#   secrets       Populate Secrets Manager placeholders interactively
#   status        Show stack status and ECS service health
#   destroy       Tear down all stacks (non-production only)
#   help          Show this message
#
# Prerequisites:
#   aws CLI v2, docker, jq, python3
#   AWS credentials with sufficient permissions
#   Set AWS_ACCOUNT_ID in environment or pass as third arg
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

ENVIRONMENT="${1:-development}"
COMMAND="${2:-all}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
STACK_PREFIX="agt-ss"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${STACK_PREFIX}-api-${ENVIRONMENT}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
CF_DIR="$(dirname "$0")/../cloudformation"
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colours
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "\n${BOLD}════════════════════════════════════════${NC}"; \
            echo -e "${BOLD} $*${NC}"; \
            echo -e "${BOLD}════════════════════════════════════════${NC}"; }

validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production) ;;
        *) error "Invalid environment: $ENVIRONMENT"; exit 1 ;;
    esac
}

validate_prerequisites() {
    for cmd in aws docker jq python3; do
        if ! command -v "$cmd" &>/dev/null; then
            error "Required command not found: $cmd"
            exit 1
        fi
    done
    aws sts get-caller-identity &>/dev/null || { error "AWS credentials not configured"; exit 1; }
    success "Prerequisites OK (aws, docker, jq, python3)"
}

# ─── CloudFormation helpers ───────────────────────────────────────────────────

deploy_stack() {
    local stack_name="$1"
    local template="$2"
    shift 2
    local params=("$@")

    info "Deploying stack: ${stack_name}"

    local param_args=()
    for p in "${params[@]}"; do
        param_args+=(--parameter-overrides "$p")
    done

    aws cloudformation deploy \
        --stack-name "$stack_name" \
        --template-file "$template" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$AWS_REGION" \
        "${param_args[@]}" \
        --tags Project=AGT-SS Environment="$ENVIRONMENT" \
        --no-fail-on-empty-changeset

    success "Stack deployed: ${stack_name}"
}

stack_output() {
    local stack_name="$1"
    local output_key="$2"
    aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='${output_key}'].OutputValue" \
        --output text
}

# ─── Stack deployments ────────────────────────────────────────────────────────

deploy_network() {
    step "01 — Network"
    deploy_stack "${STACK_PREFIX}-network-${ENVIRONMENT}" \
        "${CF_DIR}/01-network.yaml" \
        "Environment=${ENVIRONMENT}"
}

deploy_database() {
    step "02 — Database (Aurora Serverless v2)"
    deploy_stack "${STACK_PREFIX}-database-${ENVIRONMENT}" \
        "${CF_DIR}/02-database.yaml" \
        "Environment=${ENVIRONMENT}"
    success "Aurora endpoint: $(stack_output "${STACK_PREFIX}-database-${ENVIRONMENT}" ClusterEndpoint)"
}

deploy_secrets() {
    step "03 — Secrets Manager"
    deploy_stack "${STACK_PREFIX}-secrets-${ENVIRONMENT}" \
        "${CF_DIR}/03-secrets.yaml" \
        "Environment=${ENVIRONMENT}"
    warn "Secrets created with placeholder values — run './deploy.sh ${ENVIRONMENT} secrets' to populate them."
}

deploy_iam_ecr() {
    step "04 — IAM + ECR"
    local github_org="${GITHUB_ORG:-cocomgroup}"
    local github_repo="${GITHUB_REPO:-agt-ss}"
    deploy_stack "${STACK_PREFIX}-iam-ecr-${ENVIRONMENT}" \
        "${CF_DIR}/04-iam-ecr.yaml" \
        "Environment=${ENVIRONMENT}" \
        "GitHubOrg=${github_org}" \
        "GitHubRepo=${github_repo}"
    success "ECR: $(stack_output "${STACK_PREFIX}-iam-ecr-${ENVIRONMENT}" EcrRepositoryUri)"
}

deploy_compute() {
    step "05 — Compute (ECS + ALB)"
    local cert_arn="${ACM_CERT_ARN:-}"
    local alert_email="${ALERT_EMAIL:-}"
    deploy_stack "${STACK_PREFIX}-compute-${ENVIRONMENT}" \
        "${CF_DIR}/05-compute.yaml" \
        "Environment=${ENVIRONMENT}" \
        "ImageTag=${IMAGE_TAG}" \
        "CertificateArn=${cert_arn}" \
        "AlertEmail=${alert_email}"
    local alb_dns
    alb_dns=$(stack_output "${STACK_PREFIX}-compute-${ENVIRONMENT}" AlbDnsName)
    success "API available at: http://${alb_dns}"
    info    "Point your DNS CNAME → ${alb_dns}"
}

# ─── Docker build + push ──────────────────────────────────────────────────────

build_and_push() {
    step "Docker Build & Push → ${ECR_REPO}:${IMAGE_TAG}"

    info "Authenticating with ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin \
        "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    info "Building image (context: ${PROJECT_ROOT})"
    docker build \
        --file "${PROJECT_ROOT}/infra/docker/Dockerfile" \
        --tag "${ECR_REPO}:${IMAGE_TAG}" \
        --tag "${ECR_REPO}:latest" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from "${ECR_REPO}:latest" \
        "${PROJECT_ROOT}"

    info "Pushing ${ECR_REPO}:${IMAGE_TAG} ..."
    docker push "${ECR_REPO}:${IMAGE_TAG}"
    docker push "${ECR_REPO}:latest"

    success "Image pushed: ${ECR_REPO}:${IMAGE_TAG}"
}

# ─── ECS service update ───────────────────────────────────────────────────────

update_service() {
    step "ECS Service Update"
    local cluster="${STACK_PREFIX}-${ENVIRONMENT}"
    local service="${STACK_PREFIX}-api-${ENVIRONMENT}"

    info "Forcing new deployment on ${cluster}/${service} with tag ${IMAGE_TAG}..."

    # Update image in task definition
    local current_task_def
    current_task_def=$(aws ecs describe-task-definition \
        --task-definition "${STACK_PREFIX}-api-${ENVIRONMENT}" \
        --region "$AWS_REGION" \
        --query taskDefinition)

    local new_task_def
    new_task_def=$(echo "$current_task_def" | python3 -c "
import sys, json
td = json.load(sys.stdin)
for c in td.get('containerDefinitions', []):
    if c['name'] == 'api':
        parts = c['image'].split(':')
        c['image'] = parts[0] + ':${IMAGE_TAG}'
# Strip read-only fields
for f in ['taskDefinitionArn','revision','status','requiresAttributes',
          'compatibilities','registeredAt','registeredBy']:
    td.pop(f, None)
print(json.dumps(td))
")

    local new_arn
    new_arn=$(aws ecs register-task-definition \
        --region "$AWS_REGION" \
        --cli-input-json "$new_task_def" \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)

    aws ecs update-service \
        --cluster "$cluster" \
        --service "$service" \
        --task-definition "$new_arn" \
        --region "$AWS_REGION" \
        --output text > /dev/null

    info "Waiting for service stability (up to 5 min)..."
    aws ecs wait services-stable \
        --cluster "$cluster" \
        --services "$service" \
        --region "$AWS_REGION"

    success "ECS service updated to ${IMAGE_TAG}"
}

# ─── DB bootstrap ─────────────────────────────────────────────────────────────

run_db_bootstrap() {
    step "DB Bootstrap — creating schema and tables"
    local bootstrap_sql="${SCRIPT_DIR}/bootstrap.sql"
    if [[ ! -f "$bootstrap_sql" ]]; then
        warn "No bootstrap.sql found at ${bootstrap_sql} — skipping"
        return
    fi
    info "Bootstrap SQL must be run from within the VPC (e.g., a bastion or ECS exec session)"
    info "File: ${bootstrap_sql}"
    info "Run: psql -h <aurora-endpoint> -U agt_ss_admin -d agt_ss -f infra/scripts/bootstrap.sql"
}

# ─── Secrets population ───────────────────────────────────────────────────────

populate_secrets() {
    step "Populate Secrets Manager"

    populate_secret() {
        local secret_name="$1"
        local prompt="$2"
        local key="$3"

        echo ""
        echo -e "${YELLOW}Secret:${NC} ${secret_name}"
        read -r -p "${prompt}: " -s value
        echo ""

        if [[ -z "$value" ]]; then
            warn "Skipping ${secret_name} (empty value)"
            return
        fi

        local current
        current=$(aws secretsmanager get-secret-value \
            --secret-id "$secret_name" --region "$AWS_REGION" \
            --query SecretString --output text 2>/dev/null || echo "{}")

        local updated
        updated=$(echo "$current" | python3 -c "
import sys, json
d = json.load(sys.stdin)
d['${key}'] = sys.argv[1]
print(json.dumps(d))
" "$value")

        aws secretsmanager put-secret-value \
            --secret-id "$secret_name" \
            --region "$AWS_REGION" \
            --secret-string "$updated"

        success "Updated: ${secret_name}"
    }

    populate_secret \
        "agt-ss/${ENVIRONMENT}/anthropic/api-key" \
        "Enter Anthropic API key (sk-ant-...)" \
        "ANTHROPIC_API_KEY"

    echo ""
    echo -e "${YELLOW}Secret:${NC} agt-ss/${ENVIRONMENT}/api/keys"
    read -r -p "Enter API key for clients (will be set as JSON array): " api_key
    if [[ -n "$api_key" ]]; then
        aws secretsmanager put-secret-value \
            --secret-id "agt-ss/${ENVIRONMENT}/api/keys" \
            --region "$AWS_REGION" \
            --secret-string "[\"${api_key}\"]"
        success "Updated API keys"
    fi

    info "SAP and DocuSign secrets can be updated via AWS Console or CLI:"
    info "  aws secretsmanager put-secret-value --secret-id agt-ss/${ENVIRONMENT}/sap/credentials --secret-string '{...}'"
}

# ─── Status ───────────────────────────────────────────────────────────────────

show_status() {
    step "AGT-SS Stack Status — ${ENVIRONMENT}"

    for stack in network database secrets iam-ecr compute; do
        local name="${STACK_PREFIX}-${stack}-${ENVIRONMENT}"
        local status
        status=$(aws cloudformation describe-stacks \
            --stack-name "$name" --region "$AWS_REGION" \
            --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")
        if [[ "$status" == *"COMPLETE"* ]]; then
            echo -e "  ${GREEN}✓${NC} ${name}: ${status}"
        elif [[ "$status" == "NOT_FOUND" ]]; then
            echo -e "  ${YELLOW}○${NC} ${name}: not deployed"
        else
            echo -e "  ${RED}✗${NC} ${name}: ${status}"
        fi
    done

    echo ""
    local cluster="${STACK_PREFIX}-${ENVIRONMENT}"
    local service="${STACK_PREFIX}-api-${ENVIRONMENT}"
    local svc_status
    svc_status=$(aws ecs describe-services \
        --cluster "$cluster" --services "$service" \
        --region "$AWS_REGION" \
        --query "services[0].{running:runningCount,desired:desiredCount,status:status}" \
        --output json 2>/dev/null || echo "null")

    if [[ "$svc_status" != "null" ]]; then
        echo -e "  ECS Service: $(echo "$svc_status" | jq -r '"running=\(.running)/\(.desired) [\(.status)]"')"
    fi

    local alb_dns
    alb_dns=$(stack_output "${STACK_PREFIX}-compute-${ENVIRONMENT}" AlbDnsName 2>/dev/null || echo "")
    if [[ -n "$alb_dns" ]]; then
        local health
        health=$(curl -sf "http://${alb_dns}/health" 2>/dev/null | jq -r .status 2>/dev/null || echo "unreachable")
        echo -e "  API health: ${health} (http://${alb_dns}/health)"
    fi
}

# ─── Destroy ──────────────────────────────────────────────────────────────────

destroy_all() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        error "Refusing to destroy production environment. Edit this script if you really mean it."
        exit 1
    fi

    warn "This will DELETE all AGT-SS stacks for environment: ${ENVIRONMENT}"
    read -r -p "Type the environment name to confirm: " confirm
    [[ "$confirm" != "$ENVIRONMENT" ]] && { info "Aborted."; exit 0; }

    for stack in compute iam-ecr secrets database network; do
        local name="${STACK_PREFIX}-${stack}-${ENVIRONMENT}"
        if aws cloudformation describe-stacks --stack-name "$name" \
            --region "$AWS_REGION" &>/dev/null; then
            info "Deleting ${name}..."
            aws cloudformation delete-stack --stack-name "$name" --region "$AWS_REGION"
            aws cloudformation wait stack-delete-complete --stack-name "$name" --region "$AWS_REGION"
            success "Deleted: ${name}"
        fi
    done
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
    echo -e "\n${BOLD}AGT-SS Infrastructure Deployment${NC}"
    echo -e "Environment: ${CYAN}${ENVIRONMENT}${NC}  Command: ${CYAN}${COMMAND}${NC}  Region: ${CYAN}${AWS_REGION}${NC}"
    echo -e "Account: ${CYAN}${AWS_ACCOUNT_ID}${NC}  Image tag: ${CYAN}${IMAGE_TAG}${NC}\n"

    validate_environment
    validate_prerequisites

    case "$COMMAND" in
        all)
            deploy_network
            deploy_database
            deploy_secrets
            deploy_iam_ecr
            build_and_push
            deploy_compute
            run_db_bootstrap
            show_status
            ;;
        infra)
            deploy_network
            deploy_database
            deploy_secrets
            deploy_iam_ecr
            deploy_compute
            ;;
        build)
            build_and_push
            ;;
        service)
            update_service
            ;;
        secrets)
            populate_secrets
            ;;
        status)
            show_status
            ;;
        destroy)
            destroy_all
            ;;
        help|--help|-h)
            sed -n '2,/^# =\+$/p' "$0" | grep -v "^# =\+" | sed 's/^# //'
            ;;
        *)
            error "Unknown command: ${COMMAND}"
            exit 1
            ;;
    esac

    echo ""
    success "Done — ${COMMAND} completed for environment: ${ENVIRONMENT}"
}

main "$@"
